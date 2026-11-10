"""Three information shifts and matched full/stale/empty memory navigation.

All robot and memory decisions belong to the model. The driver only freezes the
experiment, starts independent episodes and scores their archived evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo import __version__
from go2_demo.complex_memory_mission import (ComplexMissionReceipt, learning_instruction,
                                            make_contract, recall_instruction)
from go2_demo.complex_world import DESTINATION_XY, final_scenario, learning_scenario
from go2_demo.provenance import git_identity, source_hashes
from go2_demo.runner import _safe_error
from scripts.run_long_memory_mission import (copy_memory, file_hash, initialize_empty,
    infrastructure_failed, memory_snapshot, phase_result, read_json, write_json)


LEARNING = ("dispatch", "fragments", "revision")
CONDITIONS = ("full", "stale", "empty")
PHASES = LEARNING + CONDITIONS
PARENTS = {"fragments": "dispatch", "revision": "fragments", "full": "revision", "stale": "fragments"}
BUDGETS = {"dispatch": {"max_model_calls": 40, "max_wall_seconds": 900, "max_sim_seconds": 150},
           "fragments": {"max_model_calls": 30, "max_wall_seconds": 900, "max_sim_seconds": 150},
           "revision": {"max_model_calls": 30, "max_wall_seconds": 900, "max_sim_seconds": 150},
           "recall": {"max_model_calls": 60, "max_wall_seconds": 1200, "max_sim_seconds": 240}}
EXPERIMENT_KIND = "complex-memory-mission-v1"


def script_hashes():
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted((ROOT / "scripts").glob("*.py"))}


def check_frozen(config):
    if (source_hashes(ROOT) != config["code_sha256"] or script_hashes() != config["scripts_sha256"]
            or git_identity(ROOT) != config["git"]
            or file_hash(ROOT / "go2_demo/prompts/memory_agent.txt") != config["memory_prompt_sha256"]):
        raise RuntimeError("Source or Git identity changed after the complex mission was frozen")


def frozen_contract(directory, config):
    path = Path(directory) / "contract.private.json"
    if file_hash(path) != config["contract_sha256"]:
        raise RuntimeError("Private contract changed after experiment freeze")
    contract = read_json(path)
    if contract["order_id"] != config["order_id"]:
        raise RuntimeError("Frozen order identifier changed")
    return contract


def worker(directory, phase, api_key_file=None):
    from go2_demo.models import APIModel
    from go2_demo.robots.registry import create_backend
    from go2_demo.runner import run_episode
    from go2_demo.task_evaluation import TaskEvaluator

    directory = Path(directory)
    run_dir = directory / phase
    backend = model = recording = None
    result, error, cleanup = None, None, []
    try:
        if phase not in PHASES:
            raise ValueError("Unknown experiment phase")
        config = read_json(directory / "experiment.json")
        check_frozen(config)
        contract = frozen_contract(directory, config)
        learning = phase in LEARNING
        scene = learning_scenario(phase) if learning else final_scenario()
        task = learning_instruction(phase, contract) if learning else recall_instruction(contract["order_id"])
        mode = "read_write" if learning else "read_only"
        model = APIModel.from_env(provider="deepseek", api_key_file=api_key_file,
            model_name="deepseek-flash", trust_env=False, timeout_s=config["api_timeout_s"],
            history_turns=6, max_output_tokens=config["max_output_tokens"])
        backend = create_backend("go2", "mujoco", asset_dir=ROOT / "assets", seed=0, scenario=scene)
        if config["recording"]:
            recording = backend.start_recording(run_dir / "third_person")
        extension = None
        if not learning:
            target = DESTINATION_XY[contract["alias_to_destination"][contract["new_alias"]]]
            backend.evaluator = TaskEvaluator(scene, target, backend._evaluation_state)
            extension = ComplexMissionReceipt(contract, backend._evaluation_state)
        print(f"Phase: {phase} | Memory: {mode} | Logs: {run_dir}", flush=True)
        result = run_episode(backend, model, mode="api", run_dir=run_dir, seed=0, task=task,
            memory_mode=mode, memory_dir=directory / "memory" / phase, history_turns=6,
            task_extension=extension, **config["budgets"][phase if learning else "recall"])
        backend = model = None
    except (Exception, KeyboardInterrupt) as exc:
        error = _safe_error(exc)
        result = {"success": False, "termination": "interrupted" if isinstance(exc, KeyboardInterrupt)
                  else "startup_error", "error": error, "is_llm_result": True}
        if not (run_dir / "metrics.json").exists():
            write_json(run_dir / "metrics.json", result)
    finally:
        operations = []
        if backend is not None:
            operations.extend((("backend.stop", backend.stop), ("backend.close", backend.close)))
        if model is not None:
            operations.append(("model.close", model.close))
        if recording is not None:
            operations.append(("recording.finish", recording.finish))
        for name, operation in operations:
            try:
                operation()
            except (Exception, KeyboardInterrupt) as exc:
                cleanup.append({"operation": name, "error": _safe_error(exc)})
        write_json(run_dir / "worker.json", {"phase": phase, "pid": os.getpid(),
            "completed_utc": datetime.now(timezone.utc).isoformat(), "cleanup_errors": cleanup, "error": error})
    return 0 if result and result.get("success") and not cleanup else 1


def launch_worker(directory, phase, api_key_file):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
               "--experiment-dir", str(directory)]
    if api_key_file:
        command.extend(("--api-key-file", str(api_key_file.resolve())))
    return subprocess.run(command, env=dict(os.environ, MUJOCO_GL="egl", TZ="Asia/Shanghai")).returncode


def prepare_memory(directory, phase):
    """Copy only actual previous agent memories; never manufacture their content."""
    if phase == "dispatch":
        return True
    destination = directory / "memory" / phase / "memory.sqlite3"
    if phase == "empty":
        initialize_empty(destination)
        return True
    source = directory / "memory" / PARENTS[phase] / "memory.sqlite3"
    if not source.exists() or not memory_snapshot(source)["content"]["records"]:
        return False
    copy_memory(source, destination)
    return True


def paired_audit(directory, contract):
    manifests = [read_json(directory / phase / "manifest.json") for phase in CONDITIONS]
    first = []
    hashes = []
    for phase in CONDITIONS:
        events = [json.loads(line) for line in (directory / phase / "events.jsonl").read_text().splitlines()]
        event = next(e for e in events if e["event"] == "model_input")
        first.append(event)
        hashes.append(file_hash(directory / phase / event["image_path"]))
    checks = {key + "_equal": all(m[key] == manifests[0][key] for m in manifests[1:])
              for key in ("task", "system_prompt_sha256", "tools_sha256", "budgets", "api_settings",
                          "robot_profile_sha256", "code_sha256")}
    public = [{k: v for k, v in e["observation"].items() if k != "agent_context"} for e in first]
    checks["initial_rgb_equal"] = len(set(hashes)) == 1
    checks["initial_public_observation_equal"] = public[0] == public[1] == public[2]
    checks["fresh_short_term_state"] = all(e["history_turns"] == 0
        and e["observation"]["agent_context"]["memory"]["working_note"] == ""
        and e["observation"]["agent_context"]["public_evidence_refs"] == [] for e in first)
    checks["all_read_only"] = all(m["memory_mode"] == "read_only" for m in manifests)
    checks["receipt_not_in_initial_observation"] = all(contract["receipt_code"] not in json.dumps(e["observation"])
                                                    for e in first)
    pids = [read_json(directory / phase / "worker.json")["pid"] for phase in PHASES]
    checks["separate_processes"] = len(set(pids)) == len(PHASES)
    for phase in PHASES:
        after = read_json(directory / phase / "memory-final.json")
        checks[phase + "_source_database_unchanged"] = after == memory_snapshot(directory / "memory" / phase / "memory.sqlite3")
        before = read_json(directory / phase / "memory-initial.json")
        if phase in PARENTS:
            checks[phase + "_exact_parent_copy"] = before == read_json(directory / PARENTS[phase] / "memory-final.json")
        if phase in CONDITIONS:
            checks[phase + "_read_only_unchanged"] = before == after
            checks[phase + "_expected_content"] = bool(before["content"]["records"]) == (phase != "empty")
        if phase == "dispatch":
            checks["initial_learning_memory_empty"] = not before["content"]["records"]
    updated = read_json(directory / "full/memory-initial.json") != read_json(directory / "stale/memory-initial.json")
    return {"checks": checks, "passed": all(checks.values()), "initial_rgb_sha256": hashes,
            "full_stale_initial_memory_different": updated,
            "note": "Private offline audit; stale is an untouched pre-revision snapshot, not hand-edited memory"}


def summarize(directory, results, contract, audit, error, stopped_reason):
    by_phase = {r["phase"]: r for r in results}
    mission_success = all(p in by_phase and by_phase[p]["metrics"].get("success") is True
                          and not infrastructure_failed(by_phase[p]) for p in LEARNING + ("full",))
    empty = by_phase.get("empty", {}).get("metrics", {})
    controls_valid = bool(not error and not stopped_reason and all(
        p in by_phase and not infrastructure_failed(by_phase[p]) for p in CONDITIONS))
    memory_dependency = bool(mission_success and audit and audit["passed"] and controls_valid
        and by_phase["full"]["retrieved_nonempty_memory"] and empty.get("termination") == "abort"
        and not empty.get("evaluation", {}).get("task_extension", {}).get("submitted"))
    stale = by_phase.get("stale", {}).get("metrics", {})
    submission = stale.get("evaluation", {}).get("task_extension", {}).get("submission") or {}
    # Attribute an outdated-answer failure only if the old answer was complete;
    # collision, missing memory or an incomplete code alone cannot prove it.
    used_complete_old_order = bool(submission
        and submission.get("order_id") == contract["order_id"]
        and submission.get("dispatch_revision") == contract["old_revision"]
        and submission.get("destination") == contract["alias_to_destination"][contract["old_alias"]]
        and submission.get("route") == contract["old_route"]
        and submission.get("receipt_code") == contract["receipt_code"])
    update_dependency = bool(mission_success and audit and audit["passed"] and controls_valid
        and audit.get("full_stale_initial_memory_different")
        and by_phase["full"]["retrieved_nonempty_memory"]
        and used_complete_old_order and stale.get("success") is False and stale.get("termination") == "finish")
    return {"experiment_kind": EXPERIMENT_KIND, "is_llm_result": True, "formal_score": None,
        "mission_success": mission_success, "demonstrated_memory_dependency": memory_dependency,
        "controls_infrastructure_valid": controls_valid,
        "demonstrated_update_dependency": update_dependency, "stale_submitted_complete_old_order": used_complete_old_order,
        "phases": results, "paired_audit": audit, "error": error, "stopped_reason": stopped_reason,
        "not_run": [p for p in PHASES if p not in by_phase],
        "api_requests": sum(r["metrics"].get("model_calls", 0) for r in results),
        "report": str(directory / "summary.json")}


def experiment(args):
    directory = args.runs_dir / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                                + "-complex-memory-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    contract = make_contract()
    write_json(directory / "contract.private.json", contract)
    config = {"schema_version": 1, "experiment_kind": EXPERIMENT_KIND, "application_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git": git_identity(ROOT),
        "code_sha256": source_hashes(ROOT), "scripts_sha256": script_hashes(),
        "memory_prompt_sha256": file_hash(ROOT / "go2_demo/prompts/memory_agent.txt"),
        "model": "deepseek-flash", "provider": "deepseek", "seed": 0,
        "budgets": BUDGETS, "history_turns": 6, "api_timeout_s": args.api_timeout,
        "max_output_tokens": args.max_output_tokens,
        "order_id": contract["order_id"], "contract_sha256": file_hash(directory / "contract.private.json"),
        "phases": list(PHASES), "memory_parents": PARENTS, "retries": 0, "formal_score": None,
        "recording": not args.no_recording,
        "design": "Three learning shifts; independent reset processes; latest/pre-revision/empty read-only final trials"}
    write_json(directory / "experiment.json", config)
    results, error, audit, stopped_reason = [], None, None, None
    try:
        for phase in PHASES:
            check_frozen(config)
            frozen_contract(directory, config)
            if not prepare_memory(directory, phase):
                stopped_reason = f"No saved agent memory for {phase}; no answer was inserted by the program"
                break
            print(f"Starting {phase}; experiment: {directory}", flush=True)
            code = launch_worker(directory, phase, args.api_key_file)
            result = phase_result(directory, phase, code)
            results.append(result)
            write_json(directory / "progress.json", {"phases": results})
            if infrastructure_failed(result):
                stopped_reason = "Infrastructure failure; no automatic retries"
                break
        if len(results) == len(PHASES):
            audit = paired_audit(directory, contract)
    except (Exception, KeyboardInterrupt) as exc:
        error = _safe_error(exc)
    summary = summarize(directory, results, contract, audit, error, stopped_reason)
    write_json(directory / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("mission_success", "demonstrated_memory_dependency",
        "demonstrated_update_dependency", "api_requests", "report")}, ensure_ascii=False), flush=True)
    return 0 if summary["mission_success"] and summary["controls_infrastructure_valid"] and audit and audit["passed"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=8192,
                        help="Freeze the per-response output budget, including reasoning")
    parser.add_argument("--api-timeout", type=float, default=90,
                        help="Freeze per-request timeout; episode deadlines still apply")
    parser.add_argument("--no-recording", action="store_true", help="Disable private control-step state archive")
    parser.add_argument("--worker", choices=PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--experiment-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_output_tokens <= 32768 or not 1 <= args.api_timeout <= 600:
        parser.error("Output tokens must be 1..32768 and API timeout 1..600 seconds")
    if args.worker:
        if args.experiment_dir is None:
            parser.error("Internal worker requires --experiment-dir")
        return worker(args.experiment_dir, args.worker, args.api_key_file)
    return experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
