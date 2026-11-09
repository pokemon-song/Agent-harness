"""Adversarial score cases: completion, denominator and comparable evidence."""

from copy import deepcopy
import json
import math

import pytest

from go2_demo.benchmark import (COMPOSITE_PROTOCOL_ID, LEGACY_PROTOCOL_ID, PROTOCOL_ID,
                               _digest, create_plan, load_plan, score_plan, compare_plans, write_json)
from go2_demo.harness import build_tools
from go2_demo.models import build_system_prompt
from go2_demo.robots.registry import get_profile
from go2_demo.scenarios import get_scenario


def plan_at(directory, label="model-a", repeats=5, protocol_id=PROTOCOL_ID):
    plan = create_plan(label=label, model_name=label, provider="compatible", repeats=repeats,
                       protocol_id=protocol_id)
    path = directory / "plan.json"
    write_json(path, plan)
    return path, plan


def record(path, plan, case, *, success=True, termination="finish", mutation=None, usage=True):
    directory = path.parent / case["run_dir"]
    task = next(t for t in plan["protocol"]["tasks"] if t["task_id"] == case["task_id"])
    protocol = plan["protocol"]
    manifest = {k: protocol[k] for k in (
        "budgets", "code_sha256", "package_versions", "asset_lock_sha256", "simulation_mode",
        "backend", "observation_protocol", "action_protocol", "completion_contract", "api_settings")}
    manifest.update({key: protocol[key] for key in
                     ("robot_id", "robot_profile", "robot_profile_sha256", "execution_contract") if key in protocol})
    manifest.update(mode="api", seed=case["seed"], task_id=case["task_id"], task=task["instruction"],
                    category=case["category"], model=plan["model"]["requested_model"], evaluation=task["evaluation"])
    evaluation = {"source": "privileged_simulator_evaluator", "goal_kind": task["evaluation"]["goal_kind"],
                  "success": success, "stopped": True, "collision": False, "fallen": False,
                  "speed_mps": .01, "distance_error_m": .05, "heading_error_rad": .1,
                  "passage_crossed": True, "beyond_exit": True}
    if task["evaluation"]["goal_kind"] == "passage":
        final_xy = [task["evaluation"]["finish_x_min_m"] + .1, 0.]
    else:
        target = get_scenario(case["task_id"]).target.xy
        distance = math.hypot(*target)
        final_xy = [v * (distance - 1.) / distance for v in target]
    evaluation["base_xy_m"] = final_xy
    metrics = {"task_id": case["task_id"], "seed": case["seed"], "mode": "api", "is_llm_result": True,
               "success": success, "termination": termination, "evaluation": evaluation,
               "final_stop": {"status": "succeeded", "speed_mps": .01, "yaw_rate_radps": .01},
               "model_calls": 1, "wall_elapsed_s": 10., "sim_elapsed_s": 5.,
               "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120} if usage else {}}
    if termination == "model_error":
        metrics["error"] = {"code": "MODEL_PROTOCOL_ERROR"}
    if mutation:
        mutation(manifest, metrics)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "tools.json", build_tools(get_profile()))
    if "robot_profile" in protocol:
        write_json(directory / "robot_profile.json", protocol["robot_profile"])
    (directory / "system_prompt.txt").write_text(build_system_prompt(get_profile()), encoding="utf-8")
    events = [{"event": "episode_start", "sim_time_s": 0., "base_xy_m": [0., 0.], "yaw_rad": 0.},
              {"event": "model_input"},
              {"event": "model_response", "model": "actual-provider-model", "latency_s": 2., "usage": metrics["usage"],
               "calls": [{"call_id": "finish-1", "name": "finish", "arguments": {}}]},
              {"event": "tool_results", "results": [{"call_id": "finish-1", "tool": "finish", "status": "succeeded"}]},
              {"event": "episode_end", "metrics": metrics}]
    (directory / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    duration = metrics["sim_elapsed_s"]
    count = math.ceil(duration / .02)
    points = [{"sim_time_s": duration * i / count,
               "base_xyz_m": [final_xy[0] * i / count, final_xy[1] * i / count, .34], "yaw_rad": 0.}
              for i in range(1, count + 1)]
    (directory / "trajectory.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in points), encoding="utf-8")
    return directory


def fill(path, plan, **kwargs):
    for case in plan["cases"]:
        record(path, plan, case, **kwargs)


def test_robot_identity_is_frozen_and_tampered_profile_invalidates_episode(tmp_path):
    path, plan = plan_at(tmp_path)
    assert plan["protocol"]["robot_id"] == "go2"
    assert "go2_demo/robots/go2/backend.py" in plan["protocol"]["code_sha256"]
    case = plan["cases"][0]
    directory = record(path, plan, case)
    changed = deepcopy(plan["protocol"]["robot_profile"])
    changed["controller_id"] = "another-controller"
    write_json(directory / "robot_profile.json", changed)
    report = score_plan(path)
    assert report["invalid_episodes"] == 1


def test_frozen_grid_has_65_trials_without_unsupported_or_baseline_tasks(tmp_path):
    path, plan = plan_at(tmp_path)
    assert len(plan["cases"]) == 65
    assert {c["seed"] for c in plan["cases"]} == {0}
    assert {c["repeat_id"] for c in plan["cases"]} == {1, 2, 3, 4, 5}
    assert set(plan["protocol"]["excluded_tasks"]) == {
        "red_marker", "crawl_red_gate2", "crawl_red_gate3", "composite_drawers_gate_b"}
    report = score_plan(path)
    assert report["score_100"] is None and report["pending_episodes"] == 65


def test_group_macro_score_and_missing_denominator_do_not_reward_cherry_picking(tmp_path):
    path, plan = plan_at(tmp_path)
    for case in plan["cases"]:
        if case["category"] == "go_to":
            record(path, plan, case)
    partial = score_plan(path)
    assert 15 < partial["provisional_score_100"] < 25
    assert partial["strict_success_score_100"] == 25
    assert partial["score_100"] is None and not partial["rankable"]
    for case in plan["cases"]:
        if case["category"] != "go_to":
            record(path, plan, case, success=False, termination="budget_exhausted")
    report = score_plan(path)
    assert report["score_100"] == partial["provisional_score_100"] and report["rankable"]
    assert report["strict_success_score_100"] == 25
    assert report["micro_success_rate"] == pytest.approx(15 / 65)


@pytest.mark.parametrize("change", [
    lambda m: m["evaluation"].update(collision=True),
    lambda m: m["evaluation"].update(fallen=True),
    lambda m: m["evaluation"].update(distance_error_m=.181),
    lambda m: m["evaluation"].update(heading_error_rad=.251),
    lambda m: m["final_stop"].update(status="failed"),
    lambda m: m.update(wall_elapsed_s=300.01),
    lambda m: m.update(sim_elapsed_s=120.01),
    lambda m: m.update(termination="budget_exhausted"),
])
def test_claimed_success_needs_independent_geometry_safety_stop_and_budget(tmp_path, change):
    path, plan = plan_at(tmp_path)
    record(path, plan, plan["cases"][0], mutation=lambda _, m: change(m))
    row = score_plan(path)["cases"][0]
    assert row["status"] == "completed" and row["strict_success"] is False


def test_passage_requires_actual_crossing_and_exit_not_standoff_distance(tmp_path):
    path, plan = plan_at(tmp_path)
    case = next(c for c in plan["cases"] if c["category"] == "go_through")
    record(path, plan, case, mutation=lambda _, m: m["evaluation"].update(passage_crossed=False))
    row = next(r for r in score_plan(path)["cases"] if r["case_id"] == case["case_id"])
    assert not row["strict_success"]
    record(path, plan, case, mutation=lambda _, m: m["evaluation"].update(distance_error_m=99.))
    row = next(r for r in score_plan(path)["cases"] if r["case_id"] == case["case_id"])
    assert row["strict_success"]  # Standoff is explicitly irrelevant to the passage contract.


def test_model_failure_is_zero_but_infrastructure_failure_invalidates_ranking(tmp_path):
    path, plan = plan_at(tmp_path)
    fill(path, plan, success=False, termination="model_error")
    assert score_plan(path)["score_100"] == 0
    record(path, plan, plan["cases"][0], success=False, termination="evaluation_error")
    report = score_plan(path)
    assert report["score_100"] is None and report["invalid_episodes"] == 1


@pytest.mark.parametrize("code", ["MODEL_HTTP_ERROR", "MODEL_TRANSPORT_ERROR", "MODEL_CONFIG_ERROR", "MODEL_ERROR"])
def test_auth_network_and_unclassified_failures_are_not_called_bad_model_planning(tmp_path, code):
    path, plan = plan_at(tmp_path)
    record(path, plan, plan["cases"][0], success=False, termination="model_error",
           mutation=lambda _, m: m.update(error={"code": code}))
    assert score_plan(path)["invalid_episodes"] == 1


@pytest.mark.parametrize("field", ["budgets", "code_sha256", "api_settings"])
def test_mixed_execution_contract_cannot_enter_official_score(tmp_path, field):
    path, plan = plan_at(tmp_path)
    record(path, plan, plan["cases"][0], mutation=lambda manifest, _: manifest.update({field: {}}))
    assert score_plan(path)["invalid_episodes"] == 1


def test_deleted_case_and_duplicate_case_are_rejected_before_scoring(tmp_path):
    path, plan = plan_at(tmp_path)
    bad = deepcopy(plan)
    bad["cases"].pop()
    write_json(path, bad)
    with pytest.raises(ValueError, match="grid"):
        score_plan(path)
    bad["cases"].append(bad["cases"][0])
    write_json(path, bad)
    with pytest.raises(ValueError, match="grid"):
        score_plan(path)


def test_missing_tokens_stay_unknown_and_inconsistent_event_archive_is_invalid(tmp_path):
    path, plan = plan_at(tmp_path)
    directory = record(path, plan, plan["cases"][0], usage=False)
    report = score_plan(path)
    assert report["diagnostics"]["known_usage"]["total_tokens"] is None
    assert report["diagnostics"]["usage_complete"] is False
    (directory / "events.jsonl").write_text("", encoding="utf-8")
    assert score_plan(path)["invalid_episodes"] == 1


def test_finish_must_actually_be_requested_and_executed_in_event_history(tmp_path):
    path, plan = plan_at(tmp_path)
    directory = record(path, plan, plan["cases"][0])
    events_path = directory / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    next(e for e in events if e["event"] == "tool_results")["results"][0]["call_id"] = "some-other-call"
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    row = score_plan(path)["cases"][0]
    assert row["status"] == "completed" and not row["strict_success"]


def test_deleted_trajectory_cannot_silently_enter_formal_score(tmp_path):
    path, plan = plan_at(tmp_path)
    directory = record(path, plan, plan["cases"][0])
    (directory / "trajectory.jsonl").unlink()
    assert score_plan(path)["invalid_episodes"] == 1


def test_missing_metrics_usage_cannot_be_marked_complete_when_events_have_usage(tmp_path):
    path, plan = plan_at(tmp_path)
    directory = record(path, plan, plan["cases"][0])
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics.pop("usage")
    write_json(directory / "metrics.json", metrics)
    events_path = directory / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[-1]["metrics"] = metrics
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    report = score_plan(path)
    assert report["diagnostics"]["known_usage"]["total_tokens"] == 120
    assert report["diagnostics"]["usage_complete"] is False


def test_compare_recomputes_evidence_and_uses_common_success_subset(tmp_path):
    a_path, a = plan_at(tmp_path / "a")
    b_path, b = plan_at(tmp_path / "b", label="model-b")
    fill(a_path, a)
    fill(b_path, b)
    record(b_path, b, b["cases"][0], success=False)
    comparison = compare_plans([a_path, b_path])
    assert comparison["rankable"]
    assert [(r["rank"], r["label"]) for r in comparison["ranking"]] == [(1, "model-a"), (2, "model-b")]
    assert len(comparison["common_success_cases"]) == 64
    assert all(m["common_success_efficiency"]["episodes"] == 64 for m in comparison["models"])
    assert comparison["models"][0]["response_models"] == ["actual-provider-model"]
    assert comparison["models"][0]["diagnostics"]["usage_complete"] is True
    c_path, _ = plan_at(tmp_path / "c", label="model-c", repeats=1)
    with pytest.raises(ValueError, match="Different frozen"):
        compare_plans([a_path, c_path])


def test_smoke_repeat_count_never_masquerades_as_standard_score(tmp_path):
    path, plan = plan_at(tmp_path, repeats=1)
    fill(path, plan)
    report = score_plan(path)
    assert report["complete"] and 60 < report["provisional_score_100"] < 100
    assert report["strict_success_score_100"] == 100
    assert not report["rankable"] and report["score_100"] is None


def test_equal_success_totals_remain_tied_despite_different_task_distribution(tmp_path):
    paths = []
    for label, counts in (("model-a", {"go_avoid_texture_cooker": 3}),
                          ("model-b", {"go_avoid_texture_cooker": 2, "go_avoid_texture_drawers": 1})):
        path, plan = plan_at(tmp_path / label, label=label)
        paths.append(path)
        for case in plan["cases"]:
            record(path, plan, case, success=case["repeat_id"] <= counts.get(case["task_id"], 0))
    ranking = compare_plans(paths)["ranking"]
    assert ranking[0]["score_100"] == ranking[1]["score_100"]
    assert [r["rank"] for r in ranking] == [1, 1]


def test_legacy_success_only_scores_remain_readable_and_cannot_mix_with_v2(tmp_path):
    old_path, old = plan_at(tmp_path / "old", protocol_id=LEGACY_PROTOCOL_ID)
    fill(old_path, old)
    report = score_plan(old_path)
    assert report["protocol_id"] == LEGACY_PROTOCOL_ID
    assert report["score_100"] == 100
    assert report["score_components"] == {"completion": 100}
    new_path, _ = plan_at(tmp_path / "new", label="new")
    with pytest.raises(ValueError, match="Different frozen"):
        compare_plans([old_path, new_path])


def test_same_success_rate_rewards_faster_shorter_successful_execution(tmp_path):
    fast_path, fast = plan_at(tmp_path / "fast", repeats=1)
    slow_path, slow = plan_at(tmp_path / "slow", label="model-b", repeats=1)
    fast_dir = record(fast_path, fast, fast["cases"][0])
    slow_dir = record(slow_path, slow, slow["cases"][0],
                      mutation=lambda _, m: m.update(wall_elapsed_s=200., sim_elapsed_s=60.))
    # Insert a measured lateral detour, keeping all timestamps and both endpoints.
    trajectory_path = slow_dir / "trajectory.jsonl"
    points = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    for i, point in enumerate(points):
        point["base_xyz_m"][1] += math.sin(math.pi * (i + 1) / len(points)) * 2.
    trajectory_path.write_text("".join(json.dumps(p) + "\n" for p in points), encoding="utf-8")
    fast_report, slow_report = score_plan(fast_path), score_plan(slow_path)
    a, b = fast_report["cases"][0], slow_report["cases"][0]
    assert a["strict_success"] and b["strict_success"]
    assert a["score_components"]["completion"] == b["score_components"]["completion"] == 60
    assert a["path_length_m"] < b["path_length_m"]
    assert a["episode_score_100"] > b["episode_score_100"]
    assert fast_report["provisional_score_100"] > slow_report["provisional_score_100"]


def test_fast_failed_episode_cannot_receive_efficiency_points(tmp_path):
    path, plan = plan_at(tmp_path)
    record(path, plan, plan["cases"][0], success=False, termination="abort",
           mutation=lambda _, m: m.update(wall_elapsed_s=.5, sim_elapsed_s=.6))
    row = score_plan(path)["cases"][0]
    assert row["status"] == "completed"
    assert row["episode_score_100"] == 0
    assert all(v == 0 for v in row["score_components"].values())


def test_v2_requires_exact_initial_anchor_and_dense_complete_trajectory(tmp_path):
    path, plan = plan_at(tmp_path)
    directory = record(path, plan, plan["cases"][0])
    events_path = directory / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events if e["event"] != "episode_start"), encoding="utf-8")
    assert score_plan(path)["invalid_episodes"] == 1
    record(path, plan, plan["cases"][0])
    trajectory_path = directory / "trajectory.jsonl"
    trajectory_path.write_text(trajectory_path.read_text().splitlines()[-1] + "\n", encoding="utf-8")
    assert score_plan(path)["invalid_episodes"] == 1


def test_unrepresentably_large_json_number_invalidates_one_trial_without_crashing_report(tmp_path):
    path, plan = plan_at(tmp_path)
    record(path, plan, plan["cases"][0], mutation=lambda _, m: m.update(wall_elapsed_s=10**400))
    report = score_plan(path)
    assert report["invalid_episodes"] == 1
    assert report["score_100"] is None
    assert report["provisional_score_100"] == 0


def test_v3_is_explicit_with_70_cases_and_five_equal_categories(tmp_path):
    path, plan = plan_at(tmp_path, protocol_id=COMPOSITE_PROTOCOL_ID)
    assert len(plan["protocol"]["tasks"]) == 14 and len(plan["cases"]) == 70
    assert plan["protocol"]["group_weights"] == {
        "go_to": .2, "distinguish": .2, "go_avoid": .2, "go_through": .2, "composite": .2}
    assert sum(c["category"] == "composite" for c in plan["cases"]) == 5
    task = next(t for t in plan["protocol"]["tasks"] if t["category"] == "composite")
    assert task["evaluation"]["cabinet_exit_x_m"] == 2.3
    assert task["evaluation"]["passage_x_m"] == 3.4
    assert task["evaluation"]["requires_stage_order"] is True
    assert task["path_reference"]["goal_kind"] == "standoff"
    assert task["path_reference"]["target_xy_m"] == [5.6, 0.]
    assert "not SPL" in plan["protocol"]["composite_path_reference"]
    assert set(score_plan(path)["groups"]) == set(plan["protocol"]["group_weights"])


def test_v3_composite_category_has_20_percent_macro_weight(tmp_path):
    path, plan = plan_at(tmp_path, protocol_id=COMPOSITE_PROTOCOL_ID)
    for case in plan["cases"]:
        record(path, plan, case, success=case["category"] == "composite",
               termination="finish" if case["category"] == "composite" else "abort")
    report = score_plan(path)
    assert report["rankable"] and report["completed_episodes"] == 70
    assert report["successful_episodes"] == 5
    assert report["strict_success_score_100"] == 20
    assert report["micro_success_rate"] == pytest.approx(5 / 70)
    assert report["score_100"] == pytest.approx(report["groups"]["composite"]["score_100"] / 5)
    assert report["score_components"]["completion"] == 12


def _rewrite_composite_trajectory(directory, waypoints, *, final_yaw=0., start_xy=(0., 0.)):
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics["evaluation"]["base_xy_m"] = list(waypoints[-1])
    write_json(directory / "metrics.json", metrics)
    events_path = directory / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[0]["base_xy_m"] = list(start_xy)
    events[-1]["metrics"] = metrics
    events_path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    duration = metrics["sim_elapsed_s"]
    count = math.ceil(duration / .02)
    vertices = [start_xy, *waypoints]
    points = []
    for i in range(1, count + 1):
        progress = i / count * (len(vertices) - 1)
        segment = min(int(progress), len(vertices) - 2)
        fraction = progress - segment
        xy = [vertices[segment][axis] + fraction * (vertices[segment + 1][axis] - vertices[segment][axis])
              for axis in (0, 1)]
        points.append({"sim_time_s": duration * i / count, "base_xyz_m": [*xy, .34], "yaw_rad": final_yaw})
    (directory / "trajectory.jsonl").write_text("".join(json.dumps(p) + "\n" for p in points), encoding="utf-8")


@pytest.mark.parametrize("route,final_yaw,start_xy", [
    ([(2.5, 1.2), (3.7, 1.2), (4.6, 0.)], 0., (0., 0.)),
    ([(4.6, 0.)], 0., (2.5, 0.)),
    ([(4.4, 0.)], 0., (0., 0.)),
    ([(4.6, 0.)], math.pi, (0., 0.)),
])
def test_v3_recomputes_stages_and_final_geometry_instead_of_trusting_success_flags(
        tmp_path, route, final_yaw, start_xy):
    path, plan = plan_at(tmp_path, repeats=1, protocol_id=COMPOSITE_PROTOCOL_ID)
    case = next(c for c in plan["cases"] if c["category"] == "composite")
    directory = record(path, plan, case,
                       mutation=lambda _, m: m["evaluation"].update(
                           stages={"avoid_drawers": True, "pass_gate": True, "approach_b": True},
                           stages_in_order=True, stage_completion_rate=1.))
    _rewrite_composite_trajectory(directory, route, final_yaw=final_yaw, start_xy=start_xy)
    row = next(r for r in score_plan(path)["cases"] if r["case_id"] == case["case_id"])
    assert row["status"] == "completed" and not row["strict_success"]
    assert not all(row["composite_evaluation"]["stages"].values())
    assert row["episode_score_100"] == 0
    assert all(value == 0 for value in row["score_components"].values())


def test_v3_valid_trace_is_recomputed_even_when_archived_stage_flags_disagree(tmp_path):
    path, plan = plan_at(tmp_path, repeats=1, protocol_id=COMPOSITE_PROTOCOL_ID)
    case = next(c for c in plan["cases"] if c["category"] == "composite")
    record(path, plan, case, mutation=lambda _, m: m["evaluation"].update(
        stages={"avoid_drawers": False, "pass_gate": False, "approach_b": False}, stages_in_order=False))
    row = next(r for r in score_plan(path)["cases"] if r["case_id"] == case["case_id"])
    assert row["strict_success"] and row["composite_evaluation"]["stages_in_order"]
    assert all(row["composite_evaluation"]["stages"].values())
    assert row["reference_distance_m"] == pytest.approx(5.6 - 1. - .18)


def test_v3_missing_final_heading_is_invalid_not_a_rewarded_shortcut(tmp_path):
    path, plan = plan_at(tmp_path, repeats=1, protocol_id=COMPOSITE_PROTOCOL_ID)
    case = next(c for c in plan["cases"] if c["category"] == "composite")
    directory = record(path, plan, case)
    trajectory_path = directory / "trajectory.jsonl"
    points = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    points[-1].pop("yaw_rad")
    trajectory_path.write_text("".join(json.dumps(p) + "\n" for p in points), encoding="utf-8")
    row = next(r for r in score_plan(path)["cases"] if r["case_id"] == case["case_id"])
    assert row["status"] == "invalid" and not row["strict_success"]


@pytest.mark.parametrize("field", ["stopped", "collision", "fallen"])
def test_v3_stage_completion_never_overrides_stop_or_safety(tmp_path, field):
    path, plan = plan_at(tmp_path, repeats=1, protocol_id=COMPOSITE_PROTOCOL_ID)
    case = next(c for c in plan["cases"] if c["category"] == "composite")
    record(path, plan, case, mutation=lambda _, m: m["evaluation"].update({field: field != "stopped"}))
    row = next(r for r in score_plan(path)["cases"] if r["case_id"] == case["case_id"])
    assert row["status"] == "completed" and not row["strict_success"]
    assert row["episode_score_100"] == 0
    if field == "stopped":
        assert not row["composite_evaluation"]["stages"]["approach_b"]


def test_v3_group_weight_tampering_is_rejected(tmp_path):
    path, plan = plan_at(tmp_path, protocol_id=COMPOSITE_PROTOCOL_ID)
    plan["protocol"]["group_weights"]["composite"] = .25
    plan["protocol_sha256"] = _digest(plan["protocol"])
    write_json(path, plan)
    with pytest.raises(ValueError, match="five categories"):
        load_plan(path)


@pytest.mark.parametrize("protocol_id", [LEGACY_PROTOCOL_ID, PROTOCOL_ID])
def test_old_protocols_remain_offline_readable_without_the_new_exclusion(tmp_path, protocol_id):
    path, plan = plan_at(tmp_path, protocol_id=protocol_id)
    plan["protocol"]["excluded_tasks"].remove("composite_drawers_gate_b")
    plan["protocol_sha256"] = _digest(plan["protocol"])
    write_json(path, plan)
    assert len(load_plan(path)["cases"]) == 65
    assert score_plan(path)["pending_episodes"] == 65


def test_v2_and_v3_are_not_cross_rankable(tmp_path):
    old, _ = plan_at(tmp_path / "v2")
    new, _ = plan_at(tmp_path / "v3", label="new", protocol_id=COMPOSITE_PROTOCOL_ID)
    with pytest.raises(ValueError, match="Different frozen protocols"):
        compare_plans([old, new])
