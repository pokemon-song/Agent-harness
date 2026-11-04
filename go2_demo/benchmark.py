"""Frozen, local model comparisons. Scoring never calls a model or the simulator."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics

from .scenarios import list_scenarios
from .evaluation_metrics import SCORING_V2, score_components, trajectory_metrics


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "go2-model-eval-v2"
LEGACY_PROTOCOL_ID = "go2-model-eval-v1"
COMPOSITE_PROTOCOL_ID = "go2-model-eval-v3"
GROUPS = ("go_to", "distinguish", "go_avoid", "go_through")
BUDGETS = {"max_model_calls": 30, "max_wall_seconds": 300,
           "max_sim_seconds": 120, "history_turns": 6}
API_SETTINGS = {"timeout_s": 90, "max_output_tokens": 8192, "thinking": {"type": "enabled"}}
INVALID_TERMINATIONS = {"configuration_error", "startup_error", "runtime_error",
                        "evaluation_error", "logging_error", "interrupted"}
FAILURE_TERMINATIONS = {"budget_exhausted", "abort", "execution_failure", "model_error"}
SCORED_MODEL_ERRORS = {"MODEL_PROTOCOL_ERROR", "MODEL_TIMEOUT"}
PROTOCOL_ERRORS = {"INVALID_ARGUMENTS", "ONE_TOOL_PER_TURN", "CALL_ID_REUSED", "EPISODE_ENDED"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _number(value, *, minimum=0):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= minimum
    except OverflowError:
        return False


def _protocol_groups(protocol_id):
    return GROUPS + ("composite",) if protocol_id == COMPOSITE_PROTOCOL_ID else GROUPS


def _tasks(*, multidimensional=True, protocol_id=PROTOCOL_ID):
    tasks = []
    for scenario in list_scenarios():
        if scenario.category not in _protocol_groups(protocol_id) or not scenario.supported:
            continue
        passage = scenario.goal_kind == "passage"
        criteria = {"goal_kind": scenario.goal_kind,
                    "standoff_m": None if passage else scenario.standoff_m,
                    "distance_tolerance_m": None if passage else scenario.distance_tolerance_m,
                    "heading_tolerance_rad": None if passage else .25,
                    "max_speed_mps": .12, "max_yaw_rate_radps": .25,
                    "requires_finish": True, "no_collision_or_fall": True}
        if passage or scenario.goal_kind == "composite":
            criteria.update(passage_x_m=scenario.passage_x,
                            passage_center_y_m=scenario.passage_center_y,
                            passage_half_width_m=scenario.passage_half_width,
                            finish_x_min_m=scenario.finish_x_min, requires_forward_crossing=True)
        if scenario.goal_kind == "composite":
            criteria.update(cabinet_exit_x_m=scenario.cabinet_exit_x,
                            requires_stage_order=True)
        task = {"task_id": scenario.task_id, "category": scenario.category,
                "instruction": scenario.instruction, "evaluation": criteria}
        if multidimensional:
            # This reference is consumed only by offline scoring, never by the model.
            task["path_reference"] = ({"goal_kind": "passage", "finish_x_min_m": scenario.finish_x_min}
                                      if passage else {"goal_kind": "standoff",
                                      "target_xy_m": list(scenario.target.xy), "standoff_m": scenario.standoff_m,
                                      "distance_tolerance_m": scenario.distance_tolerance_m})
        tasks.append(task)
    return tasks


def _cases(tasks, repeats):
    return [{"case_id": f"{task['task_id']}-r{repeat:02d}", "task_id": task["task_id"],
             "category": task["category"], "repeat_id": repeat, "seed": 0,
             "run_dir": f"episodes/{task['task_id']}/repeat-{repeat:02d}"}
            for repeat in range(1, repeats + 1) for task in tasks]


def create_plan(*, label: str, model_name: str, provider: str, repeats: int = 5,
                protocol_id: str = PROTOCOL_ID) -> dict:
    """Freeze the full denominator before any inference; never reads credentials."""
    from .harness import TOOLS
    from .models import SYSTEM_PROMPT

    if not isinstance(label, str) or not label.strip() or len(label) > 128:
        raise ValueError("label must be nonempty and at most 128 characters")
    if not isinstance(model_name, str) or not model_name.strip() or len(model_name) > 256:
        raise ValueError("model_name must be nonempty and at most 256 characters")
    if provider not in {"compatible", "deepseek", "qwen"}:
        raise ValueError("Unsupported provider")
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("repeats must be an integer from 1 to 20")
    if protocol_id not in {PROTOCOL_ID, LEGACY_PROTOCOL_ID, COMPOSITE_PROTOCOL_ID}:
        raise ValueError("Unsupported scoring protocol")
    multidimensional = protocol_id != LEGACY_PROTOCOL_ID
    packages = {}
    for name in ("mujoco", "numpy", "torch", "opencv-python", "httpx", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_recorded"
    protocol = {
        "id": protocol_id, "repeat_count": repeats, "scene_seed": 0,
        "profile": "standard" if repeats == 5 else "exploratory",
        "budgets": dict(BUDGETS), "api_settings": deepcopy(API_SETTINGS),
        "tasks": _tasks(multidimensional=multidimensional, protocol_id=protocol_id),
        "simulation_mode": "paused_during_model_inference", "backend": "MujocoGo2Backend",
        "observation_protocol": "rgb_tracked_geometry_local_odometry_v2",
        "action_protocol": "bounded_relative_motion_static_posture_v2",
        "completion_contract": "public_body_reference_and_clearance_v2",
        "code_sha256": {str(p.relative_to(ROOT)): _file_hash(p)
                        for p in sorted((ROOT / "go2_demo").glob("*.py"))},
        "package_versions": packages, "python_version": platform.python_version(),
        "asset_lock_sha256": _file_hash(ROOT / "assets.lock.json"),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "tools_sha256": _digest(TOOLS),
        "sampling": "provider defaults; no model sampling seed is requested",
        "primary_score": ("mean(group mean(task mean(success-gated weighted episode score)))"
                          if multidimensional else "100 * mean(group mean(task mean(strict success over repeats)))"),
        "failure_policy": "protocol/timeout errors score zero; HTTP/transport/config/unknown model errors invalidate",
        "excluded_tasks": [s.task_id for s in list_scenarios()
                           if s.category not in _protocol_groups(protocol_id) or not s.supported],
    }
    if multidimensional:
        protocol["scoring"] = deepcopy(SCORING_V2)
    if protocol_id == COMPOSITE_PROTOCOL_ID:
        protocol["group_weights"] = {group: .2 for group in _protocol_groups(protocol_id)}
        protocol["composite_path_reference"] = (
            "Final target standoff straight-line lower bound; ignores cabinet and gate constraints, not SPL")
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol, "protocol_sha256": _digest(protocol),
            "model": {"label": label.strip(), "requested_model": model_name.strip(), "provider": provider},
            "cases": _cases(protocol["tasks"], repeats)}


def load_plan(plan_path: Path) -> dict:
    """Keep v1/v2 reports readable while rejecting changed denominators or rules."""
    plan = _read(Path(plan_path))
    protocol = plan.get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("Invalid protocol object")
    repeats = protocol.get("repeat_count")
    if (plan.get("schema_version") != 1
            or protocol.get("id") not in {PROTOCOL_ID, LEGACY_PROTOCOL_ID, COMPOSITE_PROTOCOL_ID}
            or plan.get("protocol_sha256") != _digest(protocol)):
        raise ValueError("Invalid plan schema or protocol hash")
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("Invalid repeat count")
    multidimensional = protocol["id"] != LEGACY_PROTOCOL_ID
    if multidimensional and protocol.get("scoring") != SCORING_V2:
        raise ValueError("Plan scoring weights or normalization differ from v2/v3")
    if (protocol["id"] == COMPOSITE_PROTOCOL_ID
            and protocol.get("group_weights") != {group: .2 for group in _protocol_groups(protocol["id"])}):
        raise ValueError("Plan v3 group weights must give all five categories equal weight")
    if (protocol.get("tasks") != _tasks(multidimensional=multidimensional, protocol_id=protocol["id"])
            or protocol.get("budgets") != BUDGETS
            or protocol.get("api_settings") != API_SETTINGS or protocol.get("scene_seed") != 0
            or protocol.get("profile") != ("standard" if repeats == 5 else "exploratory")):
        raise ValueError("Plan does not follow the fixed tasks and rules")
    if plan.get("cases") != _cases(protocol["tasks"], repeats):
        raise ValueError("Plan case grid was changed, duplicated or truncated")
    model = plan.get("model", {})
    if (model.get("provider") not in {"compatible", "deepseek", "qwen"}
            or not isinstance(model.get("label"), str) or not model["label"].strip()
            or not isinstance(model.get("requested_model"), str) or not model["requested_model"].strip()):
        raise ValueError("Invalid model identity")
    return plan


def _manifest_issues(manifest, metrics, case, plan, directory):
    protocol = plan["protocol"]
    task = next(t for t in protocol["tasks"] if t["task_id"] == case["task_id"])
    expected = {"mode": "api", "seed": case["seed"], "task_id": case["task_id"],
                "task": task["instruction"], "category": case["category"],
                "model": plan["model"]["requested_model"], "evaluation": task["evaluation"]}
    for key in ("budgets", "code_sha256", "package_versions", "asset_lock_sha256",
                "simulation_mode", "backend", "observation_protocol", "action_protocol", "completion_contract"):
        expected[key] = protocol[key]
    issues = [f"manifest.{key} mismatch" for key, value in expected.items() if manifest.get(key) != value]
    if any(manifest.get("api_settings", {}).get(k) != v for k, v in protocol["api_settings"].items()):
        issues.append("API settings mismatch")
    if metrics.get("mode") != "api" or metrics.get("is_llm_result") is not True:
        issues.append("not an API model result")
    if metrics.get("task_id") != case["task_id"] or metrics.get("seed") != case["seed"]:
        issues.append("metrics task or scene seed mismatch")
    try:
        if _file_hash(directory / "system_prompt.txt") != protocol["system_prompt_sha256"]:
            issues.append("system prompt mismatch")
        tools = json.loads((directory / "tools.json").read_text(encoding="utf-8"))
        if _digest(tools) != protocol["tools_sha256"]:
            issues.append("tools mismatch")
    except (OSError, ValueError, TypeError):
        issues.append("missing or invalid prompt/tools archive")
    return issues


def _strict_success(metrics, criteria):
    evaluation, stop = metrics["evaluation"], metrics.get("final_stop", {})
    if not (metrics.get("success") is True and metrics.get("termination") == "finish"
            and evaluation.get("success") is True and evaluation.get("stopped") is True
            and evaluation.get("collision") is False and evaluation.get("fallen") is False
            and stop.get("status") == "succeeded"):
        return False
    for value, limit in ((evaluation.get("speed_mps"), .12), (stop.get("speed_mps"), .12),
                         (stop.get("yaw_rate_radps"), .25)):
        if not _number(value) or value >= limit:
            return False
    if not (1 <= metrics["model_calls"] <= BUDGETS["max_model_calls"]
            and metrics["wall_elapsed_s"] <= BUDGETS["max_wall_seconds"]
            and metrics["sim_elapsed_s"] <= BUDGETS["max_sim_seconds"]):
        return False
    if criteria["goal_kind"] == "passage":
        return evaluation.get("passage_crossed") is True and evaluation.get("beyond_exit") is True
    return (_number(evaluation.get("distance_error_m"))
            and evaluation["distance_error_m"] <= criteria["distance_tolerance_m"]
            and _number(evaluation.get("heading_error_rad"))
            and evaluation["heading_error_rad"] <= criteria["heading_tolerance_rad"])


def _composite_evidence(trajectory, start, task, *, stopped):
    """Reconstruct ordered stages and final target geometry from archived poses."""
    from .composite import evaluate_composite_trajectory

    criteria = task["evaluation"]
    anchor = {"sim_time_s": start["sim_time_s"],
              "base_xyz_m": [*start["base_xy_m"], trajectory[0]["base_xyz_m"][2]]}
    evidence = evaluate_composite_trajectory(
        [anchor, *trajectory],
        **{key: criteria[key] for key in ("cabinet_exit_x_m", "passage_x_m", "passage_center_y_m",
                                          "passage_half_width_m", "finish_x_min_m")})
    yaw = trajectory[-1].get("yaw_rad")
    if not _number(yaw, minimum=-math.inf):
        raise ValueError("Composite evaluation requires the archived final base heading")
    final_xy = trajectory[-1]["base_xyz_m"][:2]
    target_xy = task["path_reference"]["target_xy_m"]
    dx, dy = target_xy[0] - final_xy[0], target_xy[1] - final_xy[1]
    distance_error = abs(math.hypot(dx, dy) - criteria["standoff_m"])
    bearing_error = math.atan2(dy, dx) - yaw
    heading_error = abs(math.atan2(math.sin(bearing_error), math.cos(bearing_error)))
    stages = {"avoid_drawers": evidence["cabinet_cleared"] is True,
              "pass_gate": all(evidence[key] is True
                               for key in ("passage_crossed", "beyond_exit", "stages_in_order")),
              "approach_b": (stopped is True and distance_error <= criteria["distance_tolerance_m"]
                             and heading_error <= criteria["heading_tolerance_rad"])}
    return {**evidence, "stages": stages, "stage_completion_rate": sum(stages.values()) / len(stages),
            "distance_error_m": distance_error, "heading_error_rad": heading_error}


def _score_case(case, plan, root):
    row = {**case, "status": "pending", "strict_success": False, "issues": []}
    directory = root / case["run_dir"]
    if not (directory / "metrics.json").exists():
        return row
    row["status"] = "invalid"
    try:
        metrics = _read(directory / "metrics.json")
        if type(metrics.get("success")) is not bool:
            raise ValueError("Invalid success flag")
        row["termination"] = metrics.get("termination", "unknown")
        row["error_code"] = (metrics.get("error") or {}).get("code")
        if row["termination"] in INVALID_TERMINATIONS:
            row["issues"].append("infrastructure/configuration error or interrupted attempt")
            return row
        if row["termination"] == "model_error" and row["error_code"] not in SCORED_MODEL_ERRORS:
            row["issues"].append("HTTP/transport/configuration or unclassified model service error")
            return row
        if row["termination"] not in FAILURE_TERMINATIONS | {"finish"}:
            raise ValueError("Unknown termination")
        manifest = _read(directory / "manifest.json")
        row["issues"].extend(_manifest_issues(manifest, metrics, case, plan, directory))
        evaluation = metrics.get("evaluation", {})
        criteria = next(t["evaluation"] for t in plan["protocol"]["tasks"] if t["task_id"] == case["task_id"])
        if (evaluation.get("source") != "privileged_simulator_evaluator"
                or evaluation.get("goal_kind") != criteria["goal_kind"]
                or any(type(evaluation.get(k)) is not bool for k in ("success", "collision", "fallen", "stopped"))):
            row["issues"].append("missing independent evaluator evidence")
        for key in ("model_calls", "wall_elapsed_s", "sim_elapsed_s"):
            if not _number(metrics.get(key)) or (key == "model_calls" and type(metrics[key]) is not int):
                row["issues"].append(f"invalid {key}")
        if row["issues"]:
            return row
        trajectory = []
        with (directory / "trajectory.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                point = json.loads(line)
                xyz = point.get("base_xyz_m")
                if (not isinstance(xyz, list) or len(xyz) != 3
                        or not all(_number(v, minimum=-math.inf) for v in xyz)
                        or not _number(point.get("sim_time_s"))):
                    raise ValueError("Invalid archived trajectory")
                trajectory.append(point)
        if not trajectory:
            raise ValueError("Empty archived trajectory")
        row["strict_success"] = _strict_success(metrics, criteria)
        # A claimed success inconsistent with independent evidence is a failed trial.
        row.update(status="completed", collision=evaluation["collision"], fallen=evaluation["fallen"],
                   model_calls=metrics["model_calls"], wall_elapsed_s=metrics["wall_elapsed_s"],
                   sim_elapsed_s=metrics["sim_elapsed_s"], usage=metrics.get("usage", {}),
                   trajectory_points=len(trajectory))
        events = []
        with (directory / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                events.append(json.loads(line, parse_constant=lambda _: None))
        inputs = [e for e in events if e.get("event") == "model_input"]
        ends = [e for e in events if e.get("event") == "episode_end"]
        if len(inputs) != metrics["model_calls"] or len(ends) != 1 or ends[0].get("metrics") != metrics:
            raise ValueError("Event history is incomplete or disagrees with metrics")
        responses = [e for e in events if e.get("event") == "model_response"]
        row["response_models"] = sorted({e["model"] for e in responses if isinstance(e.get("model"), str)})
        row["api_latency_s"] = (sum(e["latency_s"] for e in responses)
                                if len(responses) == metrics["model_calls"]
                                and all(_number(e.get("latency_s")) for e in responses) else None)
        row["usage"] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            known = [e.get("usage", {}).get(key) for e in responses]
            row["usage"][key] = sum(v for v in known if _number(v)) if any(_number(v) for v in known) else None
        row["usage_complete"] = (len(responses) == metrics["model_calls"]
                                 and all(all(_number(e.get("usage", {}).get(k))
                                             for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                                         for e in responses)
                                 and all(_number(metrics.get("usage", {}).get(k))
                                         and metrics["usage"][k] == row["usage"][k]
                                         for k in row["usage"]))
        results = [r for e in events if e.get("event") == "tool_results" for r in e.get("results", [])]
        finish_requests = {(e.get("turn"), c.get("call_id")) for e in responses
                           for c in e.get("calls", []) if c.get("name") == "finish"}
        row["finish_executed"] = any((e.get("turn"), r.get("call_id")) in finish_requests
                                     and r.get("tool") == "finish" and r.get("status") == "succeeded"
                                     for e in events if e.get("event") == "tool_results"
                                     for r in e.get("results", []))
        row["strict_success"] &= row["finish_executed"]
        row["tool_calls"] = len(results)
        row["protocol_errors"] = sum(r.get("error_code") in PROTOCOL_ERRORS for r in results)
        if plan["protocol"]["id"] != LEGACY_PROTOCOL_ID:
            starts = [e for e in events if e.get("event") == "episode_start"]
            if len(starts) != 1:
                raise ValueError("v2/v3 requires exactly one archived episode start")
            task = next(t for t in plan["protocol"]["tasks"] if t["task_id"] == case["task_id"])
            path = trajectory_metrics(trajectory, start=starts[0], final_xy=evaluation.get("base_xy_m"),
                                      sim_elapsed_s=metrics["sim_elapsed_s"], reference=task["path_reference"])
            if criteria["goal_kind"] == "composite":
                evidence = _composite_evidence(trajectory, starts[0], task, stopped=evaluation.get("stopped"))
                row["composite_evaluation"] = evidence
                row["strict_success"] &= all(evidence["stages"].values())
            if row["strict_success"] and path["path_length_m"] < path["reference_distance_m"] - 1e-5:
                raise ValueError("Successful trajectory is shorter than the geometric distance lower bound")
            row.update(path)
            row["score_components"] = score_components(
                strict_success=row["strict_success"], path_efficiency=path["path_efficiency"],
                wall_elapsed_s=metrics["wall_elapsed_s"], sim_elapsed_s=metrics["sim_elapsed_s"],
                model_calls=metrics["model_calls"])
        else:
            row["score_components"] = {"completion": 100.0 * row["strict_success"],
                                       "total_100": 100.0 * row["strict_success"]}
        row["episode_score_100"] = row["score_components"]["total_100"]
        return row
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        row.update(status="invalid", strict_success=False)
        row["issues"].append(f"unreadable or malformed evidence: {type(exc).__name__}")
        return row


def _efficiency(rows):
    def mean(key):
        values = [r[key] for r in rows if _number(r.get(key))]
        return statistics.mean(values) if values else None
    measured = ("model_calls", "wall_elapsed_s", "sim_elapsed_s", "api_latency_s", "path_length_m",
                "reference_distance_m", "path_efficiency", "total_yaw_rad")
    return {"episodes": len(rows), "known_counts": {k: sum(_number(r.get(k)) for r in rows) for k in measured},
            "mean_model_calls": mean("model_calls"),
            "mean_wall_elapsed_s": mean("wall_elapsed_s"), "mean_sim_elapsed_s": mean("sim_elapsed_s"),
            "mean_api_latency_s": mean("api_latency_s"), "mean_path_length_m": mean("path_length_m"),
            "mean_reference_distance_m": mean("reference_distance_m"),
            "mean_path_efficiency": mean("path_efficiency"), "mean_total_yaw_rad": mean("total_yaw_rad")}


def score_plan(plan_path: Path) -> dict:
    plan_path = Path(plan_path)
    plan = load_plan(plan_path)
    rows = [_score_case(case, plan, plan_path.parent) for case in plan["cases"]]
    component_names = (tuple(SCORING_V2["weights"]) if plan["protocol"]["id"] != LEGACY_PROTOCOL_ID
                       else ("completion",))

    def components(trials):
        return {key: statistics.mean(r.get("score_components", {}).get(key, 0)
                                     if r["status"] == "completed" else 0 for r in trials)
                for key in component_names}

    groups, tasks = {}, {}
    for task in plan["protocol"]["tasks"]:
        trials = [r for r in rows if r["task_id"] == task["task_id"]]
        tasks[task["task_id"]] = {"successes": sum(r["strict_success"] for r in trials),
                                  "expected": len(trials), "success_rate": sum(r["strict_success"] for r in trials) / len(trials),
                                  "score_components": components(trials),
                                  "score_100": statistics.mean(r.get("episode_score_100", 0)
                                                              if r["status"] == "completed" else 0 for r in trials),
                                  "successful_only": _efficiency([r for r in trials if r["strict_success"]])}
    for group in _protocol_groups(plan["protocol"]["id"]):
        ids = [t["task_id"] for t in plan["protocol"]["tasks"] if t["category"] == group]
        groups[group] = {"task_ids": ids, "success_rate": statistics.mean(tasks[t]["success_rate"] for t in ids),
                         "successes": sum(tasks[t]["successes"] for t in ids),
                         "expected": sum(tasks[t]["expected"] for t in ids),
                         "score_100": statistics.mean(tasks[t]["score_100"] for t in ids),
                         "score_components": {k: statistics.mean(tasks[t]["score_components"][k] for t in ids)
                                              for k in component_names},
                         "all_completed": _efficiency([r for r in rows if r["category"] == group and r["status"] == "completed"]),
                         "successful_only": _efficiency([r for r in rows if r["category"] == group and r["strict_success"]])}
    complete = all(r["status"] != "pending" for r in rows)
    invalid = sum(r["status"] == "invalid" for r in rows)
    valid_rows = [r for r in rows if r["status"] == "completed"]
    rankable = complete and invalid == 0 and plan["protocol"]["profile"] == "standard"
    provisional = round(statistics.mean(g["score_100"] for g in groups.values()), 10)
    strict_success_score = round(100 * statistics.mean(g["success_rate"] for g in groups.values()), 10)
    component_scores = {key: round(statistics.mean(g["score_components"][key] for g in groups.values()), 10)
                        for key in component_names}
    usage = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        known = [r.get("usage", {}).get(key) for r in valid_rows]
        usage[key] = sum(v for v in known if _number(v)) if any(_number(v) for v in known) else None
    tool_calls = sum(r.get("tool_calls", 0) for r in valid_rows)
    return {"schema_version": 1, "protocol_id": plan["protocol"]["id"], "protocol_sha256": plan["protocol_sha256"],
            "created_utc": datetime.now(timezone.utc).isoformat(), "plan_created_utc": plan["created_utc"],
            "model": plan["model"], "profile": plan["protocol"]["profile"],
            "response_models": sorted({m for row in valid_rows for m in row.get("response_models", [])}),
            "complete": complete, "rankable": rankable, "expected_episodes": len(rows),
            "completed_episodes": len(valid_rows), "invalid_episodes": invalid,
            "successful_episodes": sum(r["strict_success"] for r in valid_rows),
            "failed_episodes": sum(not r["strict_success"] for r in valid_rows),
            "pending_episodes": sum(r["status"] == "pending" for r in rows),
            "score_100": provisional if rankable else None, "provisional_score_100": provisional,
            "score_components": component_scores, "strict_success_score_100": strict_success_score,
            "micro_success_rate": sum(r["strict_success"] for r in rows) / len(rows),
            "groups": groups, "tasks": tasks,
            "issues": [f"{r['case_id']}: {issue}" for r in rows for issue in r["issues"]],
            "diagnostics": {"termination_counts": dict(Counter(r.get("termination", r["status"]) for r in rows)),
                            "error_counts": dict(Counter(r["error_code"] for r in rows if r.get("error_code"))),
                            "collision_episodes": sum(r.get("collision", False) for r in valid_rows),
                            "fallen_episodes": sum(r.get("fallen", False) for r in valid_rows),
                            "all_completed": _efficiency(valid_rows),
                            "successful_only": _efficiency([r for r in valid_rows if r["strict_success"]]),
                            "known_usage": usage,
                            "usage_complete": len(valid_rows) == len(rows) and all(r.get("usage_complete") for r in valid_rows),
                            "tool_calls": tool_calls,
                            "protocol_error_rate": sum(r.get("protocol_errors", 0) for r in valid_rows) / tool_calls if tool_calls else None,
                            "currency_cost": None},
            "cases": rows}


def compare_plans(plan_paths: list[Path]) -> dict:
    if len(plan_paths) < 2:
        raise ValueError("Comparison requires at least two plans")
    paths = [Path(p).resolve() for p in plan_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate plan paths")
    reports = [score_plan(p) for p in paths]
    if len({r["protocol_sha256"] for r in reports}) != 1:
        raise ValueError("Different frozen protocols; ranking is refused")
    if len({r["model"]["label"] for r in reports}) != len(reports):
        raise ValueError("Model labels must be unique")
    common = set.intersection(*[{r["case_id"] for r in report["cases"] if r["strict_success"]} for report in reports])
    rankable = all(r["rankable"] for r in reports)
    ordered = sorted(reports, key=lambda r: -r["score_100"]) if rankable else reports
    ranking = [{"label": r["model"]["label"], "score_100": r["score_100"],
                "rank": 1 + sum(other["score_100"] > r["score_100"] for other in reports)}
               for r in ordered] if rankable else None
    return {"protocol_id": reports[0]["protocol_id"], "protocol_sha256": reports[0]["protocol_sha256"],
            "rankable": rankable, "ranking": ranking,
            "common_success_cases": sorted(common),
            "models": [{"model": report["model"], "score_100": report["score_100"],
                        "response_models": report["response_models"], "diagnostics": report["diagnostics"],
                        "expected_episodes": report["expected_episodes"],
                        "provisional_score_100": report["provisional_score_100"],
                        "strict_success_score_100": report["strict_success_score_100"],
                        "micro_success_rate": report["micro_success_rate"],
                        "successful_episodes": report["successful_episodes"],
                        "failed_episodes": report["failed_episodes"],
                        "completed_episodes": report["completed_episodes"],
                        "score_components": report["score_components"], "groups": report["groups"],
                        "pending_episodes": report["pending_episodes"], "invalid_episodes": report["invalid_episodes"],
                        "common_success_efficiency": _efficiency([r for r in report["cases"] if r["case_id"] in common])}
                       for report in ordered]}
