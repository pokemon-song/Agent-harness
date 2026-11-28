"""Run the task-independent agent with a separately configured MuJoCo task adapter."""
from copy import deepcopy
from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .general_agent import AgentLimits, GeneralAgent, READER_PROMPT, SYSTEM_PROMPT, AGENT_UPDATES_PROMPT, validate_task
from .memory_maze_agent import MazeClient


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != {"task", "environment"}:
        raise ValueError("Configuration needs exactly task and environment")
    config["task"] = validate_task(config["task"])
    env = config["environment"]
    if (not isinstance(env, dict) or set(env) - {"scenario", "floor_size_m", "checkpoint_target_ids"}
            or not isinstance(env.get("scenario"), str) or not env["scenario"]):
        raise ValueError("Environment requires a scene identifier and optional adapter settings")
    floor = env.get("floor_size_m")
    if floor is not None and (not isinstance(floor, list) or len(floor) != 2
                             or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 1 for v in floor)):
        raise ValueError("Floor dimensions must contain two finite lengths greater than one metre")
    target_ids = {t["target_id"] for t in config["task"]["targets"]}
    mapping = env.get("checkpoint_target_ids", {})
    if (not isinstance(mapping, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str) or v not in target_ids for k, v in mapping.items())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("Checkpoint mapping must use distinct public target IDs")
    return config


class MujocoTaskAdapter:
    """The ONLY new layer that knows the scene registry and private evaluation.

    It emits measured completion booleans, never next goals, coordinates, masks,
    candidate viewpoints or paths. Existing task order belongs to the benchmark.
    """
    def __init__(self, config, assets_dir, seed, output, max_wall_seconds, max_sim_seconds):
        from .backend import MujocoGo2Backend
        from .harness import Harness
        from .occluded_video import OccludedRecording
        from .scenarios import get_scenario

        self.config, self.output = deepcopy(config), Path(output)
        scene = get_scenario(config["environment"]["scenario"])
        # Shared collision policy: contacts are recorded and penalized, not terminal.
        scene = replace(scene, collision_penalty_points=5)
        if "floor_size_m" in config["environment"]:
            scene = replace(scene, floor_half_size_m=tuple(v / 2 for v in config["environment"]["floor_size_m"]))
        self.backend = MujocoGo2Backend(Path(assets_dir), seed=seed, scenario=scene, stop_on_first_collision=False)
        self.recorder = None
        try:
            self.backend.deadline_monotonic = time.monotonic() + max_wall_seconds
            self.backend.sim_deadline_s = float(self.backend.data.time) + max_sim_seconds
            self.harness = Harness(self.backend, config["task"]["instruction"])
            self.recorder = OccludedRecording(self.backend, self.output / "capture",
                                              outbound_label="GENERIC AGENT | LIVE EPISODE")
            evaluation = self.evaluation()
            mapping = config["environment"].get("checkpoint_target_ids", {})
            if evaluation.get("checkpoints") and set(mapping) != set(evaluation["checkpoints"]):
                raise ValueError("Declare a public target mapping for every benchmark checkpoint")
            if not evaluation.get("checkpoints") and mapping:
                raise ValueError("This benchmark has no named checkpoints")
        except Exception as original:
            try:
                self.close()
            except Exception as cleanup_error:
                original.add_note(f"Initialization cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}")
            raise

    def evaluation(self):
        evaluation = deepcopy(self.backend.evaluate())
        # The legacy single-target scorer folds collisions into success. Evaluate
        # the same geometric acceptance separately for the score-only policy.
        # Ordered tasks retain their independent evaluator without modification.
        if "checkpoints" not in evaluation:
            if evaluation.get("goal_kind") == "passage":
                reached = evaluation["passage_crossed"] and evaluation["beyond_exit"]
            else:
                reached = (evaluation["distance_error_m"] <= evaluation["distance_tolerance_m"]
                           and evaluation["heading_error_rad"] <= .25)
            evaluation["success"] = bool(reached and evaluation["stopped"] and not evaluation["fallen"])
        return evaluation

    def feedback(self):
        evaluation = self.evaluation()
        mapping = self.config["environment"].get("checkpoint_target_ids", {})
        milestones = [{"target_id": mapping[k], "passed": bool(row["passed"])}
                      for k, row in evaluation.get("checkpoints", {}).items()]
        if not milestones and len(self.config["task"]["targets"]) == 1:
            milestones = [{"target_id": self.config["task"]["targets"][0]["target_id"],
                           "passed": bool(evaluation["success"])}]
        return {"milestones": milestones, "task_completed": bool(evaluation["success"])}

    def close(self):
        try:
            if self.recorder:
                self.recorder.finish()
        finally:
            # A failed archive must not be retried by backend.close before it
            # releases the renderer/viewer, otherwise the same error leaks them.
            self.backend._recording = None
            self.backend.close()


class RunLog:
    def __init__(self, output):
        self.output = Path(output)
        self.stream = (self.output / "events.jsonl").open("w")
        self.adapter = None

    def __call__(self, event, **fields):
        if event == "request_started" and fields.get("role") == "planner" and self.adapter:
            frame = self.adapter.harness.frame
            path = self.output / "images" / f"request-{fields['ordinal']:04d}.png"
            path.parent.mkdir(exist_ok=True)
            Image.fromarray(frame.rgb).save(path)
            fields["image"] = {"path": str(path.relative_to(self.output)),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "image_id": frame.frame_id}
        self.stream.write(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def close(self):
        self.stream.close()


def run_episode(args):
    from .general_execution import GenericExecutor

    config = load_config(args.task_config)
    subgoal_updates = getattr(args, "subgoal_updates", "measured")
    if subgoal_updates not in {"agent", "measured"}:
        raise ValueError("subgoal_updates must be agent or measured")
    limits = AgentLimits(args.max_requests, args.max_reader_requests, args.max_consecutive_errors, args.max_wall_seconds)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "task-config.json", config)
    # Credential paths and values are intentionally absent from the public protocol.
    write(out / "protocol.json", {"architecture": "generic_subgoal_evidence_v2" if subgoal_updates == "agent" else "generic_subgoal_evidence_v1", "mode": args.mode,
          "subgoal_updates": subgoal_updates,
          "subgoal_judgment_source": "main_model" if subgoal_updates == "agent" else "measured_rules",
          "route_progress_source": "model_declared_prefix" if subgoal_updates == "agent" else "executor_measured_prefix",
          "task": config["task"], "environment": config["environment"], "model": args.model,
          "provider": args.provider, "seed": args.seed, "max_requests": args.max_requests,
          "max_reader_requests": args.max_reader_requests, "max_native_actions": args.max_native_actions,
          "max_consecutive_errors": args.max_consecutive_errors,
          "error_count_termination_enabled": args.max_consecutive_errors > 0,
          "actions_per_slice": args.actions_per_slice, "max_wall_seconds": args.max_wall_seconds,
          "max_sim_seconds": args.max_sim_seconds, "full_context_retained": True,
          "grounding": "model_selected_pixel_then_rgbd_tracking", "semantic_grounding_verified": False,
          "approach_reference": "tracked_surface_point", "task_pass_source": "independent_environment_evaluator",
          "odometry_source": "simulated_proprioception",
          "boundary_interlock": "existing_simulator_ground_truth_boundary_gate",
          "memory_updates": "throughout_episode", "automatic_stage_reset": False,
          "collision_policy": "record_and_penalize", "collision_penalty_points": 5,
          "source_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).parent.glob("general_*.py")}})
    log = RunLog(out)
    adapter = agent = planner = reader = None
    result = {"success": False, "termination": "initializing"}
    try:
        adapter = MujocoTaskAdapter(config, args.assets_dir, args.seed, out, args.max_wall_seconds, args.max_sim_seconds)
        log.adapter = adapter
        np.savez_compressed(out / "initial_state.npz", qpos=adapter.backend.data.qpos.copy(), qvel=adapter.backend.data.qvel.copy())
        options = dict(provider=args.provider, model_name=args.model, api_key_file=args.api_key_file,
                       timeout_s=90, trust_env=False, history_turns=0)
        planner = MazeClient.from_env(**options,
            system_prompt=AGENT_UPDATES_PROMPT if subgoal_updates == "agent" else SYSTEM_PROMPT,
            retain_context=True, max_output_tokens=4096)
        if args.mode == "query":
            reader = MazeClient.from_env(**options, system_prompt=READER_PROMPT, retain_context=False, max_output_tokens=2048)
        executor = GenericExecutor(adapter.harness, [t["target_id"] for t in config["task"]["targets"]],
                                   max_native_actions=args.max_native_actions, actions_per_slice=args.actions_per_slice,
                                   on_action=lambda entry: log("native_action", detail=entry))
        agent = GeneralAgent(config["task"], planner, executor, reader=reader, feedback_provider=adapter.feedback,
                             limits=limits, log=log, subgoal_updates=subgoal_updates)
        result = agent.run()
        evaluation = adapter.evaluation()
        collisions = len(adapter.backend.collision_events)
        within_budgets = (time.monotonic() <= adapter.backend.deadline_monotonic
                          and float(adapter.backend.data.time) <= adapter.backend.sim_deadline_s)
        success = bool(result["termination"] == "finish" and adapter.harness.termination == "finish"
                       and evaluation["success"] and not adapter.backend._fallen_ever and within_budgets)
        if not within_budgets and result["termination"] == "finish":
            result["termination"] = "budget_exhausted"
        checkpoints = evaluation.get("checkpoints", {})
        if checkpoints:
            order = evaluation.get("checkpoint_order", list(checkpoints))
            # Earlier achieved checkpoints retain partial credit; the final one
            # requires a properly finished successful episode, as in the benchmark.
            credited = sum(bool(checkpoints[k]["passed"]) for k in order[:-1]) + int(success)
            goal_points = 100 * credited / len(order)
        else:
            goal_points = 100 if success else 0
        result.update(success=success, collision_count=collisions, collision_penalty=5 * collisions,
                      goal_points=goal_points, score=max(0, goal_points - 5 * collisions),
                      final_sim_s=float(adapter.backend.data.time))
        write(out / "evaluation.evaluator-only.json", evaluation)
    except Exception as exc:
        result.update(termination="infrastructure_error", success=False,
                      error=f"{type(exc).__name__}: {exc}")
        if agent:
            result.update({k: v for k, v in agent.result("infrastructure_error").items() if k != "termination"})
        log("error", error=result["error"])
    finally:
        cleanup_errors = []
        def cleanup(name, operation):
            try:
                operation()
            except Exception as exc:
                cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if agent and agent.memory:
            cleanup("memory_archive", lambda: agent.memory.write(out))
        if adapter:
            # A premature finish remains failed; cleanup cannot upgrade success.
            if adapter.harness.termination is None and not adapter.backend._fallen_ever:
                cleanup("final_stop", adapter.backend.stop)
            cleanup("physical_recording", adapter.close)
            def save_trajectory():
                with (out / "trajectory.jsonl").open("w") as stream:
                    for row in adapter.backend.trajectory:
                        stream.write(json.dumps(row) + "\n")
            cleanup("trajectory", save_trajectory)
        for client in (planner, reader):
            if client:
                cleanup("model_client", client.close)
        cleanup("event_log", log.close)
        if cleanup_errors:
            result.update(termination="infrastructure_error", success=False, cleanup_errors=cleanup_errors)
        write(out / "metrics.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-config", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--mode", choices=("empty", "query"), default="query")
    p.add_argument("--subgoal-updates", choices=("agent", "measured"), default="measured",
                   help="Who judges subgoal completion, recovery and saved route progress; measured (default) uses observed execution results")
    p.add_argument("--provider", choices=("deepseek", "compatible"), default="deepseek")
    p.add_argument("--model", default="deepseek-flash")
    p.add_argument("--api-key-file", type=Path)
    p.add_argument("--assets-dir", type=Path, default=Path("assets"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-requests", type=int, default=128)
    p.add_argument("--max-reader-requests", type=int, default=16)
    p.add_argument("--max-native-actions", type=int, default=460)
    p.add_argument("--actions-per-slice", type=int, default=6)
    p.add_argument("--max-wall-seconds", type=float, default=2400.)
    p.add_argument("--max-sim-seconds", type=float, default=1880.)
    p.add_argument("--max-consecutive-errors", type=int, default=0,
                   help="Consecutive error limit; 0 (default) disables it while other budgets still apply")
    p.add_argument("--validate-only", action="store_true", help="Check configuration without simulation or API requests")
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        config = load_config(args.task_config)
        AgentLimits(args.max_requests, args.max_reader_requests, args.max_consecutive_errors, args.max_wall_seconds)
        if min(args.max_native_actions, args.actions_per_slice) < 1 or not math.isfinite(args.max_sim_seconds) or args.max_sim_seconds <= 0:
            raise ValueError("Physical budgets must be positive")
        if args.validate_only:
            from .scenarios import get_scenario
            scene = get_scenario(config["environment"]["scenario"])
            if not scene.supported:
                raise ValueError("Scene requires unsupported physical capabilities")
            print(json.dumps({"valid": True, "task": config["task"]["id"], "targets": len(config["task"]["targets"]),
                              "api_requests": 0}, ensure_ascii=False))
            return 0
        if args.output is None:
            p.error("--output is required unless --validate-only is used")
        result = run_episode(args)
    except ValueError as exc:
        p.error(str(exc))
    print(json.dumps({k: result.get(k) for k in ("success", "termination", "requests")}, ensure_ascii=False))
    return int(result["termination"] == "infrastructure_error")


if __name__ == "__main__":
    raise SystemExit(main())
