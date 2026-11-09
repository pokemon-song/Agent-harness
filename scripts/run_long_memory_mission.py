"""Cross-shift Go2 order delivery, with matched full/empty-memory recall trials.

Each phase runs in a fresh OS process and a fresh canonical simulator state.
The model chooses all robot and memory tools. This is a cross-episode mission,
not an uninterrupted physical trajectory or a statistically powered benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo import __version__
from go2_demo.memory import MemoryScope, MemoryStore
from go2_demo.memory_mission import (DESTINATION_XY, DeliveryReceipt, briefing_instruction,
                                    delivery_scenario, make_order, recall_instruction)
from go2_demo.provenance import git_identity, source_hashes
from go2_demo.robots.registry import get_profile
from go2_demo.runner import _safe_error


PHASES = ("briefing", "with_memory", "empty_memory")
BUDGETS = {"briefing": {"max_model_calls": 40, "max_wall_seconds": 900, "max_sim_seconds": 120},
           "delivery": {"max_model_calls": 30, "max_wall_seconds": 450, "max_sim_seconds": 120}}
INFRA_ERRORS = {"configuration_error", "startup_error", "runtime_error", "evaluation_error",
                "logging_error", "interrupted"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scope():
    return MemoryScope.from_profile(get_profile("go2", "mujoco"))


def memory_snapshot(path):
    store = MemoryStore(path, scope=scope(), mode="read_only", episode_id="offline-audit")
    try:
        return store.snapshot()
    finally:
        store.close()


def initialize_empty(path):
    store = MemoryStore(path, scope=scope(), mode="read_write", episode_id="empty-control-setup")
    store.close()


def copy_memory(source, destination):
    """Use SQLite backup, preserving the original experiment's database."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    dst = None
    try:
        dst = sqlite3.connect(destination)
        src.backup(dst)
    finally:
        if dst is not None:
            dst.close()
        src.close()
    if memory_snapshot(source) != memory_snapshot(destination):
        raise RuntimeError("Copied memory does not match the source snapshot")


def check_frozen(config):
    if (source_hashes(ROOT) != config["code_sha256"]
            or file_hash(__file__) != config["driver_sha256"]
            or file_hash(ROOT / "go2_demo/prompts/memory_agent.txt") != config["memory_prompt_sha256"]
            or git_identity(ROOT) != config["git"]):
        raise RuntimeError("Source or Git identity changed after the mission was frozen")


def frozen_order(directory, config):
    path = Path(directory) / "order.private.json"
    if file_hash(path) != config["order_file_sha256"]:
        raise RuntimeError("Private order changed after the mission was frozen")
    order = read_json(path)
    if order["order_id"] != config["order_id"]:
        raise RuntimeError("Order identifier disagrees with the frozen experiment")
    return order


def worker(directory, phase, api_key_file=None):
    """No prior model, simulator, note or history object enters this process."""
    from go2_demo.models import APIModel
    from go2_demo.robots.registry import create_backend
    from go2_demo.runner import run_episode
    from go2_demo.scenarios import get_scenario
    from go2_demo.task_evaluation import TaskEvaluator

    directory = Path(directory)
    run_dir = directory / phase
    backend = model = None
    cleanup_errors, worker_error = [], None
    result = None
    try:
        config = read_json(directory / "experiment.json")
        check_frozen(config)
        order = frozen_order(directory, config)
        is_briefing = phase == "briefing"
        scenario = get_scenario("composite_drawers_gate_b") if is_briefing else delivery_scenario()
        task = briefing_instruction(order) if is_briefing else recall_instruction(order["order_id"])
        extension = None if is_briefing else DeliveryReceipt(order)
        mode = "read_write" if is_briefing else "read_only"
        memory_dir = directory / "memory" / phase
        budgets = config["budgets"]["briefing" if is_briefing else "delivery"]
        model = APIModel.from_env(provider="deepseek", api_key_file=api_key_file,
                                  model_name="deepseek-flash", trust_env=False,
                                  timeout_s=90, history_turns=6, max_output_tokens=8192)
        backend = create_backend("go2", "mujoco", asset_dir=ROOT / "assets", seed=0, scenario=scenario)
        if not is_briefing:
            # The world always has identical A/B/C signs. Only this independent,
            # private scorer knows which sign the earlier public order assigned.
            backend.evaluator = TaskEvaluator(scenario, DESTINATION_XY[order["destination"]],
                                               backend._evaluation_state)
        print(f"Phase: {phase} | Memory: {mode} | Logs: {run_dir}", flush=True)
        result = run_episode(backend, model, mode="api", run_dir=run_dir, seed=0, task=task,
                             memory_mode=mode, memory_dir=memory_dir, history_turns=6,
                             task_extension=extension, **budgets)
        backend = model = None
    except (Exception, KeyboardInterrupt) as exc:
        worker_error = _safe_error(exc)
        result = {"success": False, "termination": "interrupted" if isinstance(exc, KeyboardInterrupt)
                  else "startup_error", "error": worker_error, "is_llm_result": True}
        # Cleanup can fail after run_episode archived its result. Preserve that
        # original evidence and record the worker exception separately.
        if not (run_dir / "metrics.json").exists():
            write_json(run_dir / "metrics.json", result)
    finally:
        operations = []
        if backend is not None:
            operations.extend((("backend.stop", backend.stop), ("backend.close", backend.close)))
        if model is not None:
            operations.append(("model.close", model.close))
        for name, operation in operations:
            try:
                operation()
            except (Exception, KeyboardInterrupt) as exc:
                cleanup_errors.append({"operation": name, "error": _safe_error(exc)})
        write_json(run_dir / "worker.json", {"phase": phase, "pid": os.getpid(),
                   "completed_utc": datetime.now(timezone.utc).isoformat(), "cleanup_errors": cleanup_errors,
                   "error": worker_error})
    return 0 if result and result.get("success") and not cleanup_errors else 1


def launch_worker(directory, phase, api_key_file):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase,
               "--experiment-dir", str(directory)]
    if api_key_file is not None:
        command.extend(("--api-key-file", str(api_key_file.resolve())))
    result = subprocess.run(command, env=dict(os.environ, MUJOCO_GL="egl", TZ="Asia/Shanghai"))
    return result.returncode


def phase_result(directory, phase, returncode):
    run_dir = directory / phase
    path = run_dir / "metrics.json"
    metrics = read_json(path) if path.exists() else {"success": False, "termination": "missing_result"}
    events_path = run_dir / "events.jsonl"
    results, artifact_error = [], None
    try:
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        results = [r for event in events if event["event"] == "tool_results" for r in event["results"]]
    except (OSError, ValueError, TypeError, KeyError) as exc:
        artifact_error = _safe_error(exc)
    return {"phase": phase, "returncode": returncode, "metrics": metrics,
            "artifact_error": artifact_error,
            "successful_tools": [r["tool"] for r in results if r.get("status") == "succeeded"],
            "retrieved_nonempty_memory": any(r.get("status") == "succeeded" and (
                r["tool"] == "memory_read" or r["tool"] == "memory_search" and r.get("matches")) for r in results),
            "worker": read_json(run_dir / "worker.json") if (run_dir / "worker.json").exists() else None}


def infrastructure_failed(result):
    m = result["metrics"]
    return (bool(result.get("artifact_error")) or m.get("termination") in INFRA_ERRORS | {"missing_result"}
            or result.get("worker") is None or bool(result["worker"].get("cleanup_errors"))
            or bool(result["worker"].get("error"))
            or m.get("termination") == "model_error" and (m.get("error") or {}).get("code")
            not in {"MODEL_TIMEOUT", "MODEL_PROTOCOL_ERROR"})


def paired_audit(directory, order):
    """Offline evidence, never passed back to an agent or the memory database."""
    phases = ("with_memory", "empty_memory")
    manifests = [read_json(directory / phase / "manifest.json") for phase in phases]
    initial_events = []
    image_hashes = []
    for phase in phases:
        events = [json.loads(line) for line in (directory / phase / "events.jsonl").read_text().splitlines()]
        first = next(e for e in events if e["event"] == "model_input")
        initial_events.append(first)
        image_hashes.append(file_hash(next((directory / phase / "frames").glob("000-*.png"))))
    checks = {key + "_equal": manifests[0][key] == manifests[1][key] for key in (
        "task", "system_prompt_sha256", "tools_sha256", "budgets", "api_settings", "robot_profile_sha256", "code_sha256")}
    checks["initial_rgb_equal"] = image_hashes[0] == image_hashes[1]
    public = [{k: v for k, v in e["observation"].items() if k != "agent_context"} for e in initial_events]
    checks["initial_public_observation_equal"] = public[0] == public[1]
    checks["fresh_history_and_notes"] = all(e["history_turns"] == 0 and
        e["observation"]["agent_context"]["memory"]["working_note"] == "" and
        e["observation"]["agent_context"]["public_evidence_refs"] == [] for e in initial_events)
    checks["receipt_absent_before_retrieval"] = all(order["receipt_code"] not in json.dumps(e["observation"])
                                                 for e in initial_events)
    checks["read_only_both"] = all(m["memory_mode"] == "read_only" for m in manifests)
    workers = [read_json(directory / phase / "worker.json") for phase in PHASES]
    checks["separate_worker_processes"] = len({w["pid"] for w in workers}) == 3
    seed = read_json(directory / "briefing/memory-final.json")
    checks["briefing_database_unchanged"] = seed == memory_snapshot(directory / "memory/briefing/memory.sqlite3")
    for phase in phases:
        before = read_json(directory / phase / "memory-initial.json")
        after = read_json(directory / phase / "memory-final.json")
        checks[phase + "_unchanged"] = before == after
        checks[phase + "_expected_content"] = bool(before["content"]["records"]) == (phase == "with_memory")
        if phase == "with_memory":
            checks["recall_received_exact_briefing_memory"] = before == seed
    return {"checks": checks, "passed": all(checks.values()), "initial_rgb_sha256": image_hashes,
            "note": "Private post-run audit; no evaluator data was supplied to the model"}


def experiment(args):
    directory = args.runs_dir / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                                + "-long-memory-mission-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    order = make_order()
    # The answer archive is private experiment data, never a tool or model input.
    write_json(directory / "order.private.json", order)
    config = {"schema_version": 1, "experiment_kind": "long-memory-mission-v1", "application_version": __version__,
              "created_utc": datetime.now(timezone.utc).isoformat(), "git": git_identity(ROOT),
              "code_sha256": source_hashes(ROOT), "driver_sha256": file_hash(__file__),
              "memory_prompt_sha256": file_hash(ROOT / "go2_demo/prompts/memory_agent.txt"),
              "model": "deepseek-flash", "provider": "deepseek", "seed": 0,
              "budgets": BUDGETS, "history_turns": 6, "api_timeout_s": 90, "max_output_tokens": 8192,
              "order_id": order["order_id"], "order_file_sha256": file_hash(directory / "order.private.json"),
              "phases": list(PHASES), "retries": 0, "formal_score": None,
              "design": "Cross-episode mission with fresh OS processes and canonical robot resets; read-only full/empty recall pair"}
    write_json(directory / "experiment.json", config)
    results, error, audit, stopped_reason = [], None, None, None
    try:
        for phase in PHASES:
            check_frozen(config)
            frozen_order(directory, config)
            if phase != "briefing":
                briefing_db = directory / "memory/briefing/memory.sqlite3"
                if not briefing_db.exists() or not memory_snapshot(briefing_db)["content"]["records"]:
                    stopped_reason = "The model did not leave persistent memory; no answer was inserted by the program"
                    break
                db = directory / "memory" / phase / "memory.sqlite3"
                if phase == "with_memory":
                    copy_memory(briefing_db, db)
                else:
                    initialize_empty(db)
            print(f"Starting {phase}; experiment: {directory}", flush=True)
            returncode = launch_worker(directory, phase, args.api_key_file)
            result = phase_result(directory, phase, returncode)
            results.append(result)
            write_json(directory / "progress.json", {"phases": results})
            if infrastructure_failed(result):
                stopped_reason = "Infrastructure failure; no automatic retries"
                break
        if len(results) == 3:
            audit = paired_audit(directory, order)
    except (Exception, KeyboardInterrupt) as exc:
        error = _safe_error(exc)
    by_phase = {r["phase"]: r for r in results}
    mission_success = all(p in by_phase and by_phase[p]["metrics"].get("success") is True
                          and not infrastructure_failed(by_phase[p]) for p in ("briefing", "with_memory"))
    empty = by_phase.get("empty_memory", {}).get("metrics", {})
    empty_receipt = empty.get("evaluation", {}).get("task_extension", {})
    demonstrated = bool(mission_success and audit and audit["passed"] and not error
                        and by_phase["with_memory"]["retrieved_nonempty_memory"]
                        and empty.get("termination") in {"finish", "abort"}
                        and empty_receipt.get("success") is False)
    summary = {"experiment_kind": "long-memory-mission-v1", "is_llm_result": True, "formal_score": None,
               "mission_success": mission_success, "demonstrated_memory_dependency": demonstrated,
               "phases": results, "paired_audit": audit, "error": error, "stopped_reason": stopped_reason,
               "not_run": [p for p in PHASES if p not in by_phase],
               "api_requests": sum(r["metrics"].get("model_calls", 0) for r in results),
               "report": str(directory / "summary.json")}
    write_json(directory / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("mission_success", "demonstrated_memory_dependency", "api_requests", "report")},
                     ensure_ascii=False), flush=True)
    return 0 if demonstrated else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--worker", choices=PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--experiment-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.experiment_dir is None:
            parser.error("Internal worker requires --experiment-dir")
        return worker(args.experiment_dir, args.worker, args.api_key_file)
    return experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
