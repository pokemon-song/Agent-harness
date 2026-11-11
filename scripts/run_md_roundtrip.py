"""Single continuous Go2 mission: learn a route to B, then retrace it from MD.

Only the planner context and point tracker restart at B. The same physical
backend, recording and measured odometry origin remain throughout. No route
waypoints or reversed actions are generated for the model by this driver.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo import __version__
from go2_demo.md_memory.navigation import run_md_episode
from go2_demo.md_memory.store import MDMemoryStore
from go2_demo.models import APIModel
from go2_demo.provenance import digest, git_identity, source_hashes
from go2_demo.robots.registry import create_backend
from go2_demo.roundtrip_mission import ReturnEvaluator, memory_notice, outbound_instruction, return_instruction
from go2_demo.scenarios import get_scenario

PHASES = ("outbound", "return")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hashes(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def frozen_sources():
    return {**source_hashes(ROOT), **{p.relative_to(ROOT).as_posix(): sha(p)
            for p in [*sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "go2_demo/prompts").glob("*.txt"))]}}


def safe_error(exc):
    from go2_demo.md_memory.navigation import _error
    return _error(exc)


def usage_totals(directory):
    requests = responses = complete = 0
    known = dict.fromkeys(("prompt_tokens", "completion_tokens", "total_tokens"), 0)
    errors = []
    for phase in PHASES:
        for name in ("events.jsonl", "memory-agent.jsonl"):
            path = directory / phase / name
            if not path.exists():
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError("Event must be an object")
                except (ValueError, TypeError):
                    errors.append({"file": str(path.relative_to(directory)), "line": number})
                    continue
                requests += event.get("event") in {"model_input", "model_request"}
                if event.get("event") != "model_response":
                    continue
                responses += 1
                usage = event.get("usage") or {}
                if not isinstance(usage, dict):
                    usage = {}
                valid = {key: type(usage.get(key)) is int and usage[key] >= 0 for key in known}
                complete += all(valid.values())
                for key in known:
                    if valid[key]:
                        known[key] += usage[key]
    return {"api_requests": requests, "complete_responses": responses, "known_usage": known,
            "unknown_usage_attempts": max(0, requests - complete), "accounting_errors": errors,
            "usage_complete": requests > 0 and requests == complete and not errors}


def measured_row(state):
    return {"sim_time_s": float(state.sim_time_s),
            "base_xyz_m": [float(x) for x in state.position_xyz_m], "yaw_rad": float(state.yaw_rad)}


def _validate(args):
    for key in ("outbound_calls", "return_calls", "memory_calls", "max_output_tokens"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(key + " must be positive")
    for key in ("stage_timeout", "sim_timeout", "api_timeout"):
        value = getattr(args, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(key + " must be finite and positive")


def experiment(args, *, backend_factory=None, model_factory=None):
    """Factories exist for explicit offline wiring tests, never an API score."""
    _validate(args)
    fixture = backend_factory is not None or model_factory is not None
    if fixture and (backend_factory is None or model_factory is None):
        raise ValueError("Fixture runs must provide both backend and model factories")
    scene = get_scenario("composite_drawers_gate_b")
    directory = args.runs_dir / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                                + "-md-roundtrip-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    frame_id = "roundtrip-" + uuid.uuid4().hex
    budgets = {phase: {"max_model_calls": getattr(args, phase + "_calls"),
                       "max_wall_seconds": args.stage_timeout, "max_sim_seconds": args.sim_timeout}
               for phase in PHASES}
    config = {"experiment_kind": "md-roundtrip-v1", "application_version": __version__,
              "created_utc": datetime.now(timezone.utc).isoformat(), "git": git_identity(ROOT),
              "source_sha256": frozen_sources(), "model": "fixture-not-llm" if fixture else "deepseek-flash",
              "is_llm_result": not fixture, "simulator_used": not fixture, "formal_score": None,
              "scene_id": scene.task_id, "budgets": budgets, "memory_calls": args.memory_calls,
              "api_timeout_s": args.api_timeout, "max_output_tokens": args.max_output_tokens,
              "history_turns": 2, "coordinate_frame_id": frame_id, "recording": not fixture,
              "automatic_retries": 0, "physical_world_resets_at_B": 0,
              "planner_context_reset_at_B": True, "return_memory_mode": "read_only",
              "return_tolerance_m": .30, "retrace_tolerance_m": .50,
              "outbound_task": outbound_instruction(scene), "return_task": return_instruction(),
              "memory_notice": memory_notice()}
    write_json(directory / "experiment.json", config)
    print(str(directory), flush=True)
    summary = {"experiment_kind": "md-roundtrip-v1", "is_llm_result": not fixture,
               "simulator_used": not fixture, "formal_score": None,
               "mission_success": False, "phases": [], "checks": {}, "error": None,
               "cleanup_errors": [], "return_not_run_reason": "outbound_not_completed",
               "coordinate_frame_id": frame_id, "pid": os.getpid()}
    backend = recording = planner = memory_model = None
    started = time.monotonic()
    store = MDMemoryStore(directory / "memory", scope=frame_id)
    initial = boundary = None
    return_before = None
    last_episode_returned = False

    def make_model(role):
        if fixture:
            return model_factory(role)
        return APIModel.from_env(provider="deepseek", api_key_file=args.api_key_file,
            model_name="deepseek-flash", trust_env=False, timeout_s=args.api_timeout,
            history_turns=2, max_output_tokens=args.max_output_tokens)

    try:
        backend = backend_factory(scene) if fixture else create_backend(
            "go2", "mujoco", asset_dir=ROOT / "assets", seed=0, scenario=scene)
        initial_frame = backend.capture()
        origin = tuple(float(x) for x in initial_frame.base_xy_yaw)
        if len(origin) != 3 or not all(math.isfinite(x) for x in origin):
            raise ValueError("A measured continuous odometry origin is required")
        initial = measured_row(backend._evaluation_state())
        write_json(directory / "initial-state.private.json", initial)
        if not fixture:
            recording = backend.start_recording(directory / "third_person")
        planner = make_model("outbound")
        memory_model = make_model("memory")
        print("Starting outbound", flush=True)
        metrics = run_md_episode(backend, planner, run_dir=directory / "outbound", store=store,
            memory_model=memory_model, task=config["outbound_task"], memory_notice=config["memory_notice"],
            max_memory_calls=args.memory_calls, history_turns=2, api_timeout_s=args.api_timeout,
            odometry_origin=origin, coordinate_frame_id=frame_id,
            mode="fixture" if fixture else "api", **budgets["outbound"])
        last_episode_returned = True
        planner = memory_model = None  # run_md_episode owns model cleanup.
        summary["phases"].append({"phase": "outbound", "metrics": metrics})
        if not metrics.get("success"):
            summary["return_not_run_reason"] = "outbound_failed"
        elif not store.snapshot()["files"] or store.snapshot()["pending_critical_events"]:
            summary["return_not_run_reason"] = "published_memory_unavailable"
        else:
            if frozen_sources() != config["source_sha256"] or git_identity(ROOT) != config["git"]:
                raise RuntimeError("Source or Git changed during experiment")
            boundary = measured_row(backend._evaluation_state())
            outbound = [initial, *deepcopy(backend.trajectory)]
            return_index = len(backend.trajectory)
            write_json(directory / "turnaround.private.json", {
                "state": boundary, "return_start_index": return_index,
                "world_reset": False, "planner_history_reset": True, "coordinate_frame_id": frame_id,
                "shared_odometry_origin": list(origin), "memory_snapshot": store.snapshot()})
            return_before = tree_hashes(store.root)
            backend.evaluator = ReturnEvaluator(backend._evaluation_state, outbound, origin[:2],
                boundary["base_xyz_m"][:2], return_index, scene)
            summary["checks"]["same_physical_boundary"] = measured_row(backend._evaluation_state()) == boundary
            planner = make_model("return")
            last_episode_returned = False
            print("Starting return (same world, fresh planner, read-only memory)", flush=True)
            metrics = run_md_episode(backend, planner, run_dir=directory / "return", store=store,
                read_only=True, task=config["return_task"], max_memory_calls=args.memory_calls,
                history_turns=2, api_timeout_s=args.api_timeout, odometry_origin=origin,
                coordinate_frame_id=frame_id, require_memory_for_motion=True,
                mode="fixture" if fixture else "api", **budgets["return"])
            last_episode_returned = True
            planner = None
            summary["phases"].append({"phase": "return", "metrics": metrics})
            summary["return_not_run_reason"] = None
            summary["checks"]["return_memory_unchanged"] = tree_hashes(store.root) == return_before
            inputs = [json.loads(line) for line in (directory / "return/events.jsonl").read_text().splitlines()
                      if json.loads(line).get("event") == "model_input"]
            first = inputs[0] if inputs else None
            summary["checks"]["fresh_return_history"] = first is not None and first["history_turns"] == 0
            summary["checks"]["shared_coordinate_frame"] = bool(first and
                first["observation"]["agent_context"].get("coordinate_frame_id") == frame_id and
                first["observation"]["local_odometry"]["reference"] == "mission_start_body_horizontal")
            summary["checks"]["same_simulation_clock"] = bool(first and
                first["observation"]["sim_time_s"] == boundary["sim_time_s"])
            summary["checks"]["return_read_memory"] = metrics.get("memory_read_calls", 0) > 0
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = safe_error(exc)
        for phase in PHASES:
            path = directory / phase / "metrics.json"
            if path.exists() and not any(row["phase"] == phase for row in summary["phases"]):
                summary["phases"].append({"phase": phase, "metrics": read_json(path)})
    finally:
        if backend is not None and not last_episode_returned:
            try:
                backend.stop()
            except BaseException as exc:
                summary["cleanup_errors"].append({"operation": "stop", **safe_error(exc)})
        if backend is not None:
            try:
                summary["final_state"] = measured_row(backend._evaluation_state())
                with (directory / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
                    for row in ([initial] if initial else []) + backend.trajectory:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                summary["sim_elapsed_s"] = summary["final_state"]["sim_time_s"] - initial["sim_time_s"]
            except BaseException as exc:
                summary["cleanup_errors"].append({"operation": "archive_state", **safe_error(exc)})
        for name, instance in (("planner", planner), ("memory", memory_model), ("backend", backend)):
            if instance is not None:
                try:
                    instance.close()
                except BaseException as exc:
                    summary["cleanup_errors"].append({"operation": name + ".close", **safe_error(exc)})
        if recording is not None:
            try:
                recording.finish()
            except BaseException as exc:
                summary["cleanup_errors"].append({"operation": "recording.finish", **safe_error(exc)})
        try:
            summary["checks"]["source_frozen"] = (frozen_sources() == config["source_sha256"]
                                                    and git_identity(ROOT) == config["git"])
        except BaseException as exc:
            summary["checks"]["source_frozen"] = False
            summary["cleanup_errors"].append({"operation": "source_check", **safe_error(exc)})
        if return_before is not None:
            try:
                summary["checks"]["return_memory_unchanged"] = tree_hashes(store.root) == return_before
            except BaseException as exc:
                summary["checks"]["return_memory_unchanged"] = False
                summary["cleanup_errors"].append({"operation": "memory_hash", **safe_error(exc)})
        try:
            summary.update(usage_totals(directory))
        except BaseException as exc:
            summary.update(api_requests=None, complete_responses=None, known_usage={}, usage_complete=False,
                           unknown_usage_attempts=None, accounting_errors=[safe_error(exc)])
        summary["wall_elapsed_s"] = time.monotonic() - started
        try:
            summary["memory_final"] = store.snapshot()
        except BaseException as exc:
            summary["memory_final"] = None
            summary["cleanup_errors"].append({"operation": "memory_snapshot", **safe_error(exc)})
        summary["path_length_m"] = sum(row["metrics"].get("path_length_m") or 0 for row in summary["phases"])
        summary["mission_success"] = bool(len(summary["phases"]) == 2 and
            all(row["metrics"].get("success") for row in summary["phases"]) and
            len(summary["checks"]) >= 7 and all(summary["checks"].values()) and
            not summary["error"] and not summary["cleanup_errors"] and not summary["accounting_errors"])
        write_json(directory / "summary.json", summary)
    print(json.dumps({"directory": str(directory), "mission_success": summary["mission_success"],
                      "api_requests": summary["api_requests"], "return_not_run_reason": summary["return_not_run_reason"]}), flush=True)
    return directory, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--api-key-file", type=Path, default=ROOT.parent / "deepseekapi.txt")
    parser.add_argument("--stage-timeout", type=float, default=1800., help="Wall seconds for each leg, including memory processing")
    parser.add_argument("--sim-timeout", type=float, default=300., help="Simulation seconds per leg")
    parser.add_argument("--outbound-calls", type=int, default=60)
    parser.add_argument("--return-calls", type=int, default=90)
    parser.add_argument("--memory-calls", type=int, default=48)
    parser.add_argument("--api-timeout", type=float, default=180.)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    args = parser.parse_args(argv)
    _, summary = experiment(args)
    return 0 if summary["mission_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
