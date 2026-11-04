"""Summarize recorded model-comparison failures offline, without changing scores.

Only completed/invalid rows in each report snapshot are inspected. Pending
episodes, response text, reasoning, tool arguments and credentials are not read
as diagnostic output. Error codes describe recorded outcomes, not their causes.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import tempfile


MODELS = ("deepseek", "qwen")
TERMINATIONS = {"finish", "abort", "budget_exhausted", "execution_failure", "model_error",
                "configuration_error", "startup_error", "runtime_error", "evaluation_error",
                "logging_error", "interrupted"}
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:+/-]{1,200}\Z")
ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,100}\Z")
ERROR_DETAIL_MESSAGES = {
    "Model response was truncated; no tool will execute.": "response_truncated",
    "The model returned invalid or non-finite JSON.": "json_decoding_rejected",
}


def _reject_constant(_value):
    raise ValueError("Non-finite JSON value")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _json(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)


def _object(path, issues, kind):
    try:
        value = _json(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Expected object")
        return value
    except (OSError, ValueError, UnicodeError, RecursionError):
        issues.append(f"{kind}_missing_or_malformed")
        return None


def _number(value, *, integer=False):
    try:
        return (type(value) in ((int,) if integer else (int, float))
                and math.isfinite(value) and value >= 0)
    except OverflowError:
        return False


def _code(value, issues):
    if value is None or value == "":
        return None
    if isinstance(value, str) and ERROR_CODE.fullmatch(value):
        return value
    issues.append("unrecognized_error_code")
    return "UNRECOGNIZED_ERROR_CODE"


def _error_detail(error, code):
    """Expose only exact internal message categories, never message contents."""
    if code == "MODEL_TIMEOUT":
        return "model_timeout"
    if code == "MODEL_PROTOCOL_ERROR" and isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return ERROR_DETAIL_MESSAGES.get(message, "unspecified")
    return "unspecified"


def _numeric(primary, fallback, key, issues, *, integer=False):
    value = primary.get(key) if key in primary else fallback.get(key)
    if value is None:
        return None
    if _number(value, integer=integer):
        return value
    issues.append(f"invalid_{key}")
    return None


def _boolean(primary, fallback, key, issues):
    value = primary.get(key) if key in primary else fallback.get(key)
    if value is None or type(value) is bool:
        return value
    issues.append(f"invalid_{key}")
    return None


def _events(path):
    issues, codes, turns = [], Counter(), set()
    declared = succeeded = False
    tool_results = unattributed = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = _json(line)
                    if not isinstance(event, dict):
                        raise ValueError("Expected event object")
                except (ValueError, RecursionError):
                    issues.append(f"events_invalid_json_line_{line_number}")
                    continue
                if event.get("event") == "model_response":
                    calls = event.get("calls", [])
                    if not isinstance(calls, list) or any(not isinstance(c, dict) for c in calls):
                        issues.append(f"events_invalid_calls_line_{line_number}")
                    else:
                        declared |= any(c.get("name") == "finish" for c in calls)
                if event.get("event") != "tool_results":
                    continue
                results = event.get("results")
                if not isinstance(results, list):
                    issues.append(f"events_invalid_results_line_{line_number}")
                    continue
                has_error = False
                for result in results:
                    if not isinstance(result, dict):
                        issues.append(f"events_invalid_result_line_{line_number}")
                        continue
                    tool_results += 1
                    code = _code(result.get("error_code"), issues)
                    if code is not None:
                        codes[code] += 1
                    status = result.get("status")
                    if status is not None and not isinstance(status, str):
                        issues.append(f"events_invalid_status_line_{line_number}")
                    has_error |= code is not None or isinstance(status, str) and status in {"failed", "timed_out"}
                    succeeded |= result.get("tool") == "finish" and result.get("status") == "succeeded"
                if has_error:
                    turn = event.get("turn")
                    if _number(turn, integer=True):
                        turns.add(turn)
                    else:
                        unattributed += 1
                        issues.append(f"events_missing_error_turn_line_{line_number}")
    except (OSError, UnicodeError):
        issues.append("events_missing_or_unreadable")
    complete = not issues
    return {"finish_declared": True if declared else (False if complete else None),
            "finish_tool_succeeded": True if succeeded else (False if complete else None),
            "tool_error_code_counts": dict(sorted(codes.items())),
            "tool_error_turns": len(turns) if complete or turns else None,
            "known_tool_error_turns": len(turns), "tool_results": tool_results,
            "unattributed_error_events": unattributed, "events_complete": complete, "issues": issues}


def _case(model_dir, case, row, task):
    issues = []
    strict_success = row.get("strict_success")
    if type(strict_success) is not bool:
        issues.append("invalid_report_strict_success")
    result = {"case_id": case["case_id"], "task_id": case.get("task_id"),
              "repeat_id": case.get("repeat_id"), "status": row["status"],
              "strict_success": strict_success if type(strict_success) is bool else None,
              "error_detail_kind": "unspecified", "issues": issues}
    relative = case.get("run_dir")
    try:
        if not isinstance(relative, str):
            raise ValueError("Invalid directory")
        path = Path(relative)
        directory = (model_dir / path).resolve()
        if path.is_absolute() or ".." in path.parts or not directory.is_relative_to(model_dir) or directory == model_dir:
            raise ValueError("Directory outside the model experiment")
    except (ValueError, OSError, RuntimeError):
        issues.append("unsafe_or_invalid_case_directory")
        return result
    result["raw_directory"] = (Path(model_dir.name) / path).as_posix()
    if row.get("run_dir") != relative:
        issues.append("report_plan_directory_mismatch")
    metrics = _object(directory / "metrics.json", issues, "metrics") or {}
    evaluation = metrics.get("evaluation", {})
    if not isinstance(evaluation, dict):
        issues.append("invalid_evaluation_object")
        evaluation = {}
    termination = metrics.get("termination", row.get("termination"))
    result["termination"] = termination if isinstance(termination, str) and termination in TERMINATIONS else None
    if termination is not None and result["termination"] is None:
        issues.append("unrecognized_termination")
    if metrics.get("termination") is not None and row.get("termination") is not None and metrics["termination"] != row["termination"]:
        issues.append("metrics_report_termination_mismatch")
    error = metrics.get("error")
    if error is not None and not isinstance(error, dict):
        issues.append("invalid_error_object")
    result["error_code"] = _code(error.get("code") if isinstance(error, dict) else row.get("error_code"), issues)
    result["error_detail_kind"] = _error_detail(error, result["error_code"])
    for key in ("model_calls", "wall_elapsed_s", "sim_elapsed_s"):
        result[key] = _numeric(metrics, row, key, issues, integer=key == "model_calls")
    result["path_length_m"] = _numeric(evaluation, row, "path_length_m", issues)
    for key in ("collision", "fallen"):
        result[key] = _boolean(evaluation, row, key, issues)
    evaluation_success = _boolean(evaluation, {}, "success", issues)
    metrics_success = _boolean(metrics, {}, "success", issues)
    result["evaluation_success"] = evaluation_success
    result["metrics_success"] = metrics_success
    result["physical_acceptance_passed_but_metrics_unsuccessful"] = (
        evaluation_success is True and metrics_success is False
        if evaluation_success is not None and metrics_success is not None else None)
    goal_kind = task.get("evaluation", {}).get("goal_kind") if isinstance(task.get("evaluation"), dict) else None
    result["goal_kind"] = goal_kind
    if evaluation.get("goal_kind") is not None and evaluation["goal_kind"] != goal_kind:
        issues.append("evaluation_plan_goal_kind_mismatch")
    if goal_kind == "standoff":
        for key in ("distance_to_target_center_m", "distance_error_m", "heading_error_rad"):
            result[key] = _numeric(evaluation, {}, key, issues)
    elif goal_kind == "passage":
        for key in ("passage_crossed", "beyond_exit"):
            result[key] = _boolean(evaluation, {}, key, issues)
    else:
        issues.append("missing_plan_goal_kind")
    stop = metrics.get("final_stop")
    stop_status = stop.get("status") if isinstance(stop, dict) else None
    result["final_stop_status"] = stop_status if isinstance(stop_status, str) and stop_status in {"succeeded", "failed"} else None
    if stop_status is not None and result["final_stop_status"] is None:
        issues.append("invalid_final_stop_status")
    events = _events(directory / "events.jsonl")
    issues.extend(events.pop("issues"))
    result.update(events)
    return result


def _model(experiment_dir, name):
    directory = (experiment_dir / name).resolve()
    issues = []
    result = {"model_directory": name, "issues": issues, "failure_cases": [], "error_detail_counts": {}}
    if not directory.is_relative_to(experiment_dir):
        issues.append("model_directory_outside_experiment")
        result.update(analysis_complete=False, formal_ready=False, counts=None)
        return result
    plan = _object(directory / "plan.json", issues, "plan")
    report = _object(directory / "report.json", issues, "report")
    if plan is None or report is None or not isinstance(plan.get("cases"), list) or not isinstance(report.get("cases"), list):
        if not issues:
            issues.append("invalid_case_lists")
        result.update(analysis_complete=False, formal_ready=False, counts=None)
        return result
    protocol = plan.get("protocol")
    if not isinstance(protocol, dict):
        issues.append("invalid_plan_protocol")
        protocol = {}
    model = plan.get("model", {})
    result["model"] = {key: value if isinstance(value, str) and IDENTIFIER.fullmatch(value) else None
                       for key, value in model.items() if key in {"label", "requested_model", "provider"}} if isinstance(model, dict) else {}
    snapshot = report.get("created_utc")
    result["report_created_utc"] = snapshot if isinstance(snapshot, str) and re.fullmatch(r"[0-9TZ:+.\-]{1,80}", snapshot) else None
    result["protocol_sha256"] = plan.get("protocol_sha256") if isinstance(plan.get("protocol_sha256"), str) and re.fullmatch(r"[a-f0-9]{64}", plan["protocol_sha256"]) else None
    if report.get("protocol_sha256") != plan.get("protocol_sha256"):
        issues.append("report_plan_protocol_mismatch")
    cases = {}
    for case in plan["cases"]:
        if (not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
                or not re.fullmatch(r"[a-z0-9_]+-r[0-9]+", case["case_id"])
                or not isinstance(case.get("task_id"), str)
                or not re.fullmatch(r"[a-z0-9_]+", case["task_id"])
                or not _number(case.get("repeat_id"), integer=True)):
            issues.append("invalid_plan_case")
            continue
        if case["case_id"] in cases:
            issues.append("duplicate_plan_case")
        cases[case["case_id"]] = case
    task_list = protocol.get("tasks", [])
    tasks = {t["task_id"]: t for t in task_list if isinstance(t, dict) and isinstance(t.get("task_id"), str)} if isinstance(task_list, list) else {}
    counts = {"expected": len(cases), "successful": 0, "valid_failed": 0, "invalid": 0,
              "pending": 0, "unclassified": 0, "unrepresented": 0}
    seen, details = set(), []
    for row in report["cases"]:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or row["case_id"] not in cases:
            issues.append("unknown_or_malformed_report_case")
            continue
        case_id = row["case_id"]
        if case_id in seen:
            issues.append("duplicate_report_case")
            continue
        seen.add(case_id)
        status = row.get("status")
        if status == "pending":
            counts["pending"] += 1
            continue
        if status == "invalid":
            counts["invalid"] += 1
        elif status == "completed" and type(row.get("strict_success")) is bool:
            counts["successful" if row["strict_success"] else "valid_failed"] += 1
        else:
            counts["unclassified"] += 1
            issues.append(f"{case_id}:invalid_report_status_or_success")
            continue
        case = cases[case_id]
        detail = _case(directory, case, row, tasks.get(case["task_id"], {}))
        details.append(detail)
        if status == "invalid" or row.get("strict_success") is False:
            result["failure_cases"].append(detail)
        issues.extend(f"{case_id}:{issue}" for issue in detail["issues"])
    counts["unrepresented"] = len(cases) - len(seen)
    if counts["unrepresented"]:
        issues.append("report_does_not_cover_entire_plan")
    counts["recorded"] = len(details)
    result["counts"] = counts
    result["termination_counts"] = dict(sorted(Counter(d["termination"] for d in details if d.get("termination") is not None).items()))
    result["error_counts"] = dict(sorted(Counter(d["error_code"] for d in details if d.get("error_code") is not None).items()))
    result["error_detail_counts"] = dict(sorted(Counter(d["error_detail_kind"] for d in result["failure_cases"]).items()))
    for name in ("collision", "fallen"):
        known = [d[name] for d in details if type(d.get(name)) is bool]
        result[f"{name}_cases"] = sum(known)
        result[f"{name}_known_cases"] = len(known)
        result[f"{name}_unknown_cases"] = len(details) - len(known)
    code_counts = Counter()
    for detail in details:
        code_counts.update(detail.get("tool_error_code_counts", {}))
    result["tool_error_code_counts"] = dict(sorted(code_counts.items()))
    result["known_tool_error_turns"] = sum(d.get("known_tool_error_turns", 0) for d in details)
    result["tool_events_complete"] = all(d.get("events_complete") is True for d in details)
    result["tool_error_turns"] = result["known_tool_error_turns"] if result["tool_events_complete"] else None
    result["tool_events_complete_cases"] = sum(d.get("events_complete") is True for d in details)
    result["unattributed_tool_error_events"] = sum(d.get("unattributed_error_events", 0) for d in details)
    result["finish_declared_but_strict_failed_case_ids"] = [d["case_id"] for d in details if d.get("finish_declared") is True and d.get("strict_success") is False]
    result["finish_declared_but_strict_failed_cases"] = len(result["finish_declared_but_strict_failed_case_ids"])
    result["finish_declaration_known_cases"] = sum(type(d.get("finish_declared")) is bool for d in details)
    accepted = [d["case_id"] for d in details if d.get("physical_acceptance_passed_but_metrics_unsuccessful") is True]
    result["physical_acceptance_passed_but_metrics_unsuccessful"] = {
        "label": "独立物理验收通过但未形成严格成功", "case_count": len(accepted), "case_ids": accepted,
        "known_cases": sum(type(d.get("physical_acceptance_passed_but_metrics_unsuccessful")) is bool for d in details),
    }
    result["analysis_complete"] = not issues
    result["formal_ready"] = (not issues and counts["expected"] == 65
                              and counts["successful"] + counts["valid_failed"] == 65
                              and counts["invalid"] == counts["pending"] == counts["unclassified"] == 0
                              and report.get("complete") is True and report.get("rankable") is True
                              and protocol.get("profile") == "standard" and protocol.get("repeat_count") == 5)
    return result


def analyze(experiment_dir: Path):
    experiment_dir = experiment_dir.expanduser().resolve(strict=True)
    if not experiment_dir.is_dir():
        raise ValueError("Experiment directory is not a directory")
    models = {name: _model(experiment_dir, name) for name in MODELS}
    issues = []
    hashes = {model.get("protocol_sha256") for model in models.values()}
    if None in hashes or len(hashes) != 1:
        issues.append("model_protocols_missing_or_different")
    provisional = bool(issues) or not all(model.get("formal_ready") is True for model in models.values())
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "provisional": provisional, "status": "provisional" if provisional else "complete",
            "scope": "Only completed/invalid rows in the saved report snapshots; pending cases are not inspected.",
            "tool_error_turn_definition": "A distinct recorded turn with a tool error_code or failed/timed_out tool result; counts are known evidence only.",
            "attribution_limit": "Error codes and timeouts are observed outcomes. Raw API responses are not archived; this analysis does not infer planning ability or the layer responsible for JSON/protocol failures.",
            "scoring_unchanged": True, "issues": issues, "models": models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.experiment_dir)
    destination = args.experiment_dir.expanduser().resolve() / "failure-analysis.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent,
                                         prefix=".failure-analysis-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(destination), "provisional": report["provisional"],
                      "counts": {name: model.get("counts") for name, model in report["models"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
