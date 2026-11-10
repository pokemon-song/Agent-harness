"""Repeat the existing six-phase complex task with independent Markdown memory.

Scene geometry and acceptance rules are unchanged. Agent architecture, memory
budget and history differ, so this is an independent diagnostic, not a matched
causal estimate of improvement over the previous single-agent experiment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo import __version__
from go2_demo.complex_memory_mission import ComplexMissionReceipt, learning_instruction, make_contract, recall_instruction
from go2_demo.complex_world import DESTINATION_XY, learning_scenario, final_scenario
from go2_demo.md_memory.store import MDMemoryStore
from go2_demo.provenance import git_identity, source_hashes, digest
from scripts.run_complex_memory_mission import BUDGETS, LEARNING, CONDITIONS, PHASES, PARENTS


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def frozen_sources():
    return {**source_hashes(ROOT), **{p.relative_to(ROOT).as_posix(): sha(p)
        for p in [*sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "go2_demo/prompts").glob("*.txt"))]}}


def check_frozen(directory, config):
    if frozen_sources() != config["source_sha256"] or git_identity(ROOT) != config["git"]:
        raise RuntimeError("Source or Git changed after experiment freeze")
    if sha(Path(directory) / "contract.private.json") != config["contract_sha256"]:
        raise RuntimeError("Contract changed after experiment freeze")


def prepare_memory(directory, phase, scope):
    destination = directory / "memory" / phase
    if phase in {"dispatch", "empty"}:
        if destination.exists():
            raise FileExistsError("Refusing to reuse a phase memory directory")
        MDMemoryStore(destination, scope=scope)
        return True
    source = directory / "memory" / PARENTS[phase]
    if not source.exists():
        return False
    parent = MDMemoryStore(source, scope=scope)
    snapshot = parent.snapshot()
    if not snapshot["files"] or (phase in CONDITIONS and snapshot["pending_critical_events"]):
        return False
    before = tree_hashes(source)
    shutil.copytree(source, destination)
    if tree_hashes(source) != before or tree_hashes(destination) != before:
        raise RuntimeError("Memory copy is not an exact preserved parent snapshot")
    return True


def worker(directory, phase, api_key_file):
    from go2_demo.models import APIModel
    from go2_demo.md_memory.navigation import run_md_episode
    from go2_demo.robots.registry import create_backend
    from go2_demo.task_evaluation import TaskEvaluator

    directory = Path(directory)
    config = read_json(directory / "experiment.json")
    check_frozen(directory, config)
    contract = read_json(directory / "contract.private.json")
    run_dir, memory_dir = directory / phase, directory / "memory" / phase
    run_dir.mkdir(parents=True, exist_ok=False)
    backend = planner = memory_model = recording = None
    metrics, cleanup, error = {}, [], None
    episode_returned = False
    before = tree_hashes(memory_dir)
    try:
        learning = phase in LEARNING
        scene = learning_scenario(phase) if learning else final_scenario()
        # Only the memory agent receives this shift's dispatch notice. Navigation
        # planning gets the same physical task, without a duplicate answer channel.
        task = scene.instruction if learning else recall_instruction(contract["order_id"])
        notice = learning_instruction(phase, contract) if learning else None
        store = MDMemoryStore(memory_dir, scope=config["memory_scope"])
        kwargs = dict(provider="deepseek", api_key_file=api_key_file, model_name="deepseek-flash",
                      trust_env=False, history_turns=config["history_turns"],
                      timeout_s=config["api_timeout_s"], max_output_tokens=config["max_output_tokens"])
        planner = APIModel.from_env(**kwargs)
        if learning:
            memory_model = APIModel.from_env(**kwargs)
        backend = create_backend("go2", "mujoco", asset_dir=ROOT / "assets", seed=0, scenario=scene)
        if config["recording"]:
            recording = backend.start_recording(run_dir / "third_person")
        extension = None
        if not learning:
            destination = DESTINATION_XY[contract["alias_to_destination"][contract["new_alias"]]]
            backend.evaluator = TaskEvaluator(scene, destination, backend._evaluation_state)
            extension = ComplexMissionReceipt(contract, backend._evaluation_state)
        metrics = run_md_episode(backend, planner, run_dir=run_dir, store=store, memory_model=memory_model,
            read_only=not learning, task=task, memory_notice=notice, task_extension=extension,
            max_memory_calls=config["memory_calls_per_learning_phase"], history_turns=config["history_turns"],
            api_timeout_s=config["api_timeout_s"], **config["budgets"][phase if learning else "recall"])
        episode_returned = True
        planner = memory_model = None  # episode owns and closes both model roles
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "code": getattr(exc, "code", "STARTUP_ERROR")}
        if (run_dir / "metrics.json").exists():
            metrics = read_json(run_dir / "metrics.json")
        else:
            metrics = {"success": False, "termination": "startup_error", "error": error, "is_llm_result": True}
            write_json(run_dir / "metrics.json", metrics)
    finally:
        operations = []
        if backend is not None and not episode_returned:
            operations.append(("backend.stop", backend.stop))
        if recording is not None:
            operations.append(("recording.finish", recording.finish))
        if backend is not None:
            operations.append(("backend.close", backend.close))
        for name, instance in (("planner", planner), ("memory_model", memory_model)):
            if instance is not None:
                operations.append((name + ".close", instance.close))
        for name, close in operations:
            try:
                close()
            except BaseException as exc:
                cleanup.append({"operation": name, "type": type(exc).__name__})
        after = tree_hashes(memory_dir)
        write_json(run_dir / "worker.json", {"phase": phase, "pid": os.getpid(), "error": error,
            "cleanup_errors": cleanup, "memory_before_digest": digest(before), "memory_after_digest": digest(after),
            "readonly_memory_unchanged": phase in LEARNING or before == after})
    return 0 if metrics.get("success") and not error and not cleanup and (phase in LEARNING or before == after) else 1


def model_accounting(directory):
    requests, responses, complete = 0, 0, 0
    errors = []
    known = {k: 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    for phase in PHASES:
        # The navigation runner uses events.jsonl for planner decisions and a
        # separate memory-agent log for its worker, with no duplicated accounting.
        for path in (directory / phase / "events.jsonl", directory / phase / "memory-agent.jsonl"):
            if not path.exists():
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError("Expected an event object")
                except (ValueError, TypeError):
                    errors.append({"path": path.relative_to(directory).as_posix(), "line": number,
                                   "code": "INVALID_EVENT_JSON"})
                    continue
                requests += item.get("event") in {"model_input", "model_request"}
                if item.get("event") != "model_response":
                    continue
                responses += 1
                usage = item.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                valid = [type(usage.get(k)) is int and usage[k] >= 0 for k in known]
                complete += all(valid)
                for key, ok in zip(known, valid):
                    if ok:
                        known[key] += usage[key]
    return {"api_requests": requests, "complete_responses": responses, "known_usage": known,
            "complete_usage_responses": complete, "accounting_errors": errors,
            "usage_complete": requests > 0 and requests == complete and not errors,
            "unknown_usage_attempts": max(0, requests - complete)}


def final_audit(directory, results):
    by_phase = {r["phase"]: r for r in results}
    if not all(p in by_phase for p in CONDITIONS):
        return None
    manifests = [read_json(directory / p / "manifest.json") for p in CONDITIONS]
    checks = {k + "_equal": all(k in m and m[k] is not None and m[k] == manifests[0].get(k) for m in manifests)
              for k in ("task", "system_prompt_sha256", "tools_sha256", "budgets", "api_settings", "robot_profile_sha256", "code_sha256")}
    first = []
    for phase in CONDITIONS:
        events = [json.loads(line) for line in (directory / phase / "events.jsonl").read_text().splitlines()]
        first.append(next(e for e in events if e["event"] == "model_input"))
    checks["same_initial_rgb"] = len({sha(directory / p / e["image_path"]) for p, e in zip(CONDITIONS, first)}) == 1
    checks["fresh_histories"] = all(e.get("history_turns") == 0 for e in first)
    public = []
    for event in first:
        observation = dict(event["observation"])
        for key in ("memory", "memory_snapshot"):
            observation.pop(key, None)
        context = dict(observation.get("agent_context", {}))
        for key in ("episode_id", "remaining_wall_s"):
            context.pop(key, None)
        observation["agent_context"] = context
        public.append(observation)
    checks["same_public_initial_observation"] = all(value == public[0] for value in public)
    checks["read_only_integrity"] = all(by_phase[p]["worker"].get("readonly_memory_unchanged") for p in CONDITIONS)
    checks["separate_processes"] = len({r["worker"]["pid"] for r in results}) == len(results)
    checks["exact_parent_lineage"] = all(
        by_phase[child]["worker"]["memory_before_digest"] == by_phase[parent]["worker"]["memory_after_digest"]
        for child, parent in PARENTS.items())
    checks["archived_memory_unchanged"] = all(
        digest(tree_hashes(directory / "memory" / row["phase"])) == row["worker"]["memory_after_digest"]
        for row in results)
    controls_memory = [event["observation"]["memory"] for event in first]
    checks["fresh_memory_read_state"] = all(value.get("read_receipts") == [] and value.get("required_reads") == []
        and value.get("needs_refresh") is False for value in controls_memory)
    checks["controls_read_only_no_pending_critical"] = all(value.get("mode") == "read_only"
        and value.get("pending_critical_events") == [] for value in controls_memory)
    empty_store = MDMemoryStore(directory / "memory/empty", scope=controls_memory[-1]["scope"])
    empty_snapshot = empty_store.snapshot()
    checks["empty_control_is_empty"] = not empty_snapshot["files"] and empty_snapshot["latest_event_seq"] == 0
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "Configuration/read-only/initial-frame checks; semantic memories are not automatically verified"}


def experiment(args):
    directory = args.runs_dir / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                                + "-md-complex-memory-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    contract = make_contract()
    write_json(directory / "contract.private.json", contract)
    config = {"experiment_kind": "md-complex-memory-v1", "application_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git": git_identity(ROOT),
        "source_sha256": frozen_sources(), "contract_sha256": sha(directory / "contract.private.json"),
        "memory_scope": "world-" + contract["order_id"], "order_id": contract["order_id"],
        "model": "deepseek-flash", "provider": "deepseek", "seed": 0, "budgets": BUDGETS,
        "history_turns": 2, "api_timeout_s": args.api_timeout, "max_output_tokens": args.max_output_tokens,
        "memory_calls_per_learning_phase": args.memory_calls, "recording": True,
        "phases": list(PHASES), "memory_parents": PARENTS, "automatic_retries": 0,
        "is_llm_result": True, "formal_score": None,
        "comparison_limit": "Original geometry/scoring preserved; roles, memory budgets, prompts and history differ"}
    write_json(directory / "experiment.json", config)
    print(f"Experiment: {directory}", flush=True)
    results, stopped_reason, error = [], None, None
    try:
        for phase in PHASES:
            check_frozen(directory, config)
            if not prepare_memory(directory, phase, config["memory_scope"]):
                stopped_reason = f"No published Markdown or unresolved critical source events for {phase}; no answer inserted"
                break
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--experiment-dir", str(directory)]
            if args.api_key_file:
                command.extend(["--api-key-file", str(args.api_key_file.resolve())])
            print(f"Starting {phase}", flush=True)
            code = subprocess.run(command, env=dict(os.environ, MUJOCO_GL="egl")).returncode
            row = {"phase": phase, "returncode": code, "metrics": read_json(directory / phase / "metrics.json"),
                   "worker": read_json(directory / phase / "worker.json")}
            results.append(row)
            write_json(directory / "progress.json", {"phases": results})
            if (row["worker"].get("error") or row["worker"]["cleanup_errors"] or not row["worker"]["readonly_memory_unchanged"]
                    or row["metrics"].get("termination") in {"startup_error", "runtime_error", "interrupted", "logging_error"}
                    or row["metrics"].get("memory_agent_stopped") is False):
                stopped_reason = "Runtime, cleanup or integrity failure; no automatic restart"
                break
    except (Exception, KeyboardInterrupt) as exc:
        error = {"type": type(exc).__name__, "code": getattr(exc, "code", "DRIVER_ERROR")}
    by_phase = {r["phase"]: r for r in results}
    audit = None
    if all(p in by_phase for p in CONDITIONS):
        try:
            audit = final_audit(directory, results)
        except Exception as exc:
            audit = {"passed": False, "error": {"type": type(exc).__name__, "code": "AUDIT_INCOMPLETE"}}
    success = all(p in by_phase and by_phase[p]["metrics"].get("success")
                  and by_phase[p]["metrics"].get("termination") == "finish"
                  and by_phase[p]["metrics"].get("memory_agent_stopped") is True
                  and not by_phase[p]["metrics"].get("cleanup_errors")
                  and by_phase[p]["returncode"] == 0
                  and not by_phase[p]["worker"].get("error")
                  and by_phase[p]["worker"].get("readonly_memory_unchanged") is True
                  and not by_phase[p]["worker"].get("cleanup_errors")
                  for p in (*LEARNING, "full"))
    summary = {"experiment_kind": config["experiment_kind"], "is_llm_result": True, "formal_score": None,
        "mission_success": bool(success and not error and not any(r["worker"]["cleanup_errors"] for r in results)),
        "phases": results, "not_run": [p for p in PHASES if p not in by_phase],
        "stopped_reason": stopped_reason, "error": error, "paired_audit": audit,
        "demonstrated_memory_dependency": False, "demonstrated_update_dependency": False,
        "dependency_note": "No causal dependency conclusion is auto-awarded; inspect full/stale/empty receipt and read evidence",
        **model_accounting(directory)}
    write_json(directory / "summary.json", summary)
    print(json.dumps({"mission_success": summary["mission_success"], "api_requests": summary["api_requests"],
                      "not_run": summary["not_run"], "directory": str(directory)}, ensure_ascii=False), flush=True)
    return 0 if summary["mission_success"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--api-timeout", type=float, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--memory-calls", type=int, default=12)
    parser.add_argument("--worker", choices=PHASES, help=argparse.SUPPRESS)
    parser.add_argument("--experiment-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.api_timeout <= 180 or not 1 <= args.max_output_tokens <= 16384 or not 1 <= args.memory_calls <= 30:
        parser.error("Invalid response/memory budget")
    if args.worker:
        if args.experiment_dir is None:
            parser.error("Worker needs an experiment directory")
        return worker(args.experiment_dir, args.worker, args.api_key_file)
    return experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
