"""Run independent live A-to-B episodes. No replay, archive, or checkpoint inputs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import math
import multiprocessing
from pathlib import Path
import time
from uuid import uuid4

from .backend import MujocoGo2Backend
from .fresh_episode_memory import FreshEpisodeMemory, FreshMemoryReader
from .fresh_episode_planning import CONTACT_RULE, FreshEpisodePlanner, public_depth_sectors
from .harness import Harness
from .memory_perception import TaskLabelRGBD, task_targets
from .models import ModelError, _usage_counters
from .occluded_return import TASK_ID, write
from .occluded_video import OccludedRecording
from .paired_return_run import ACTION_LIMIT, API_TIMEOUT, MODES, SIM_LIMIT, WALL_LIMIT, ReturnFollower
from .perception import PointTracker
from .return_route_control import _End
from .runner import RunLog, image_data_url, json_text
from .scenarios import get_scenario


TASK = f"先找到蓝球 A 并合格停靠，再返回绿方块 B 并合格停靠。共用起点里程计坐标。{CONTACT_RULE}"


def episode_scene(floor_size_m=None):
    scene = get_scenario(TASK_ID)
    if floor_size_m is None:
        return scene
    length, width = floor_size_m
    if not all(math.isfinite(value) and value > 0 for value in (length, width)):
        raise ValueError("Floor length and width must be finite positive distances.")
    return replace(scene, floor_half_size_m=(length / 2, width / 2))


class StageFollower(ReturnFollower):
    def __init__(self, harness, log, memory, stage, *, max_actions=ACTION_LIMIT):
        super().__init__(harness, log, max_actions=max_actions)
        self.target_label = stage
        self.memory = memory
        self.on_action = lambda action: log.write("native_action", stage=stage, detail=action)

    def act(self, name, **args):
        if self.target_label == "A" and name == "finish":
            raise _End("replan", "A_stop_not_yet_confirmed")
        reply = super().act(name, **args)
        if self.target_label == "A":
            self.memory.observe(self.harness, len(self.actions))
            # Only the milestone bit crosses this boundary. No private geometry goes to the planner.
            if self.harness.backend._ordered_evaluator.a_witness is not None:
                raise _End("handoff", "A_stop_confirmed")
        return reply


def play_stage(stage, mode, harness, log, memory, planner, cfg):
    follower = StageFollower(harness, log, memory, stage, max_actions=cfg.max_native_actions)
    detector = TaskLabelRGBD(task_targets(TASK_ID))
    backend = harness.backend
    t0, wall_start = harness.frame.sim_time_s, time.monotonic()
    backend.deadline_monotonic = wall_start + cfg.max_wall_seconds
    backend.sim_deadline_s = t0 + cfg.max_sim_seconds
    path_start = tuple(harness.frame.base_xy_yaw[:2])
    traj_start = len(backend.trajectory)
    history, usage, choices = [], Counter(), Counter()
    planned = reads = consecutive_errors = unknown_usage = 0
    stop, detail = "budget_exhausted", "shared_api_limit"
    first_target = None
    while planned + reads < cfg.max_requests:
        remaining = backend.deadline_monotonic - time.monotonic()
        if remaining <= 0 or harness.frame.sim_time_s >= backend.sim_deadline_s:
            detail = "wall_or_sim_limit"
            break
        obs, frame = harness.observation(), harness.frame
        obs["target_candidates"] = detector.inspect(obs, frame.rgb, frame.depth_m, frame.intrinsics)[0]["target_candidates"]
        if stage == "A":
            obs["depth_sectors"] = public_depth_sectors(frame)
        if first_target is None and any(c["task_label"] == stage for c in obs["target_candidates"]):
            first_target = {"sim_time_s": frame.sim_time_s, "planner_decision": planned+1, "image_id": frame.frame_id}
        planned += 1
        log.write("planner_input", stage=stage, index=planned, image=log.save_frame(frame, f"{stage}-{planned:03d}"),
                  memory_card=memory.card, history_turns=min(len(history), 6), current_odometry=obs["local_odometry"],
                  current_targets=obs["target_candidates"], depth_sectors=obs.get("depth_sectors"))
        try:
            call = planner.choose(obs, image_data_url(frame.rgb), memory.card, history, stage=stage,
                                  timeout_s=min(API_TIMEOUT, remaining))
        except ModelError as exc:
            consecutive_errors += 1
            log.write("planner_error", stage=stage, index=planned, code=exc.code, finish_reason=planner.response_finish_reason,
                      detail=str(exc) if exc.code == "MODEL_PROTOCOL_ERROR" else None, usage=_usage_counters(planner.raw_usage))
            if consecutive_errors >= cfg.max_consecutive_errors:
                stop, detail = "model_error", exc.code
                break
            history.append({"tool": "previous_request", "status": "error", "error_code": exc.code})
            continue
        finally:
            counters = _usage_counters(planner.raw_usage)
            usage.update(counters)
            unknown_usage += int("total_tokens" not in counters)
        consecutive_errors = 0
        name, args = call.name, call.arguments
        choices[name] += 1
        log.write("model_choice", stage=stage, index=planned, tool=name, arguments=args, usage=counters)
        if name == "memory_query":
            answer = memory.handle(args["question"], timeout_s=min(API_TIMEOUT, backend.deadline_monotonic-time.monotonic()),
                                   remaining_requests=cfg.max_requests-planned-reads)
            reads += answer["requests"]
            unknown_usage += answer["unknown_usage_requests"]
            usage.update(answer["known_usage"])
            log.write("memory_result", stage=stage, detail=answer)
            feedback = {"status": answer["status"], "error_code": answer["error"],
                        "returned_waypoints": len((answer["card"] or {}).get("waypoints", []))}
        elif name == "abort":
            stop, detail = "abort", args["reason"]
            break
        else:
            feedback = follower.execute(name, args, planned)
        history.append({"tool": name, "arguments": args, "status": feedback["status"], "feedback": feedback})
        log.write("planner_feedback", stage=stage, index=planned, detail=feedback)
        if planned % 10 == 0 or name == "follow_waypoints" or feedback["status"] == "handoff":
            print(f"{mode} {stage}: {planned}+{reads} requests, {name} -> {feedback['status']}, {len(follower.actions)} native", flush=True)
        if feedback["status"] in {"handoff", "finish", "collision", "fall", "budget_exhausted"}:
            stop, detail = feedback["status"], feedback.get("error_code")
            break
    path = [path_start, *[r["base_xyz_m"][:2] for r in backend.trajectory[traj_start:]]]
    return {"termination": stop, "detail": detail, "stage": stage, "start_sim_s": t0, "end_sim_s": harness.frame.sim_time_s,
            "wall_seconds": time.monotonic()-wall_start, "path_m": sum(math.dist(a,b) for a,b in zip(path,path[1:])),
            "planner_requests": planned, "reader_requests": reads, "total_requests": planned+reads,
            "known_usage": dict(usage), "unknown_usage_requests": unknown_usage, "first_planner_target_candidate": first_target,
            "planner_tools": dict(choices), "native_attempts": dict(Counter(a["tool"] for a in follower.actions)),
            "boundary_refusals": sum(a["reply"].get("error_code") == "BOUNDARY_BLOCKED" for a in follower.actions),
            "depth_refusals": follower.refusals}


def run_episode(cfg, mode):
    scene = episode_scene(cfg.floor_size_m)
    out = cfg.runs_dir / mode
    out.mkdir(exist_ok=False, parents=True)
    backend = MujocoGo2Backend(cfg.assets_dir, seed=cfg.seed, scenario=scene,
                               stop_on_first_collision=True, continue_on_b_contact=True)
    recorder = planner = reader = None
    log = RunLog(out)
    try:
        recorder = OccludedRecording(backend, out/"video", outbound_label=f"LIVE SEARCH FOR A | {mode.upper()}",
                                     return_label=f"{mode.upper()} | LIVE RETURN TO B")
        harness = Harness(backend, TASK)
        episode_id = str(uuid4())
        write(out/"manifest.json", {"episode_id": episode_id, "mode": mode, "seed": cfg.seed, "task": TASK_ID,
              "outbound": "fresh_live_planner", "return": "same_episode_live_planner",
              "floor_size_m": [2 * half for half in scene.floor_half_size_m],
              "archive_inputs": [], "initial_memory": "empty", "initial_odometry": harness.observation()["local_odometry"],
              "model": "deepseek-flash", "thinking": False, "api_limit_per_stage": cfg.max_requests,
              "max_consecutive_planner_errors": cfg.max_consecutive_errors, "max_native_per_stage": cfg.max_native_actions,
              "max_wall_s_per_stage": cfg.max_wall_seconds, "max_sim_s_per_stage": cfg.max_sim_seconds, "api_timeout_s": API_TIMEOUT,
              "max_output_tokens": 8192, "history": 6, "reset_model_history_at_A": True,
              "first_contact_terminates": False, "completion_requirements": backend.public_task_spec})
        options = dict(provider="deepseek", model_name="deepseek-flash", api_key_file=cfg.api_key_file,
                       timeout_s=API_TIMEOUT, max_output_tokens=8192, trust_env=False, history_turns=0)
        planner = FreshEpisodePlanner.from_env(**options)
        if mode == "query":
            reader = FreshMemoryReader.from_env(**options)
        memory = FreshEpisodeMemory(mode, TASK_ID, reader)
        assert memory.card is None
        memory.observe(harness, 0)
        print(f"{mode}: live A search from reset simulator", flush=True)
        outbound = play_stage("A", mode, harness, log, memory, planner, cfg)
        stages = {"A": outbound}
        if outbound["termination"] == "handoff":
            recorder.mark_handoff()
            memory.freeze()
            log.write("A_handoff", episode_id=episode_id, sim_time_s=harness.frame.sim_time_s,
                      current_odometry=harness.observation()["local_odometry"])
            harness.tracker = PointTracker()
            harness.tracker.update(harness.frame)
            harness.cache.clear()
            stages["B"] = play_stage("B", mode, harness, log, memory, planner, cfg)
        final_stage = stages.get("B", outbound)
        stop = final_stage["termination"]
        if mode != "empty":
            write(out/"episode_documents.json", memory.documents())
        final_stop = backend.stop()
        frame = backend.capture()
        log.save_frame(frame, "final")
        log.write_contacts(backend)
        evaluation = backend.evaluate()
        if backend.collision_terminated:
            stop = "collision"
        elif backend._fallen_ever:
            stop = "fall"
        elif time.monotonic() > backend.deadline_monotonic or frame.sim_time_s > backend.sim_deadline_s:
            stop = "budget_exhausted"
        result = {"episode_id": episode_id, "mode": mode, "termination": stop, "detail": final_stage["detail"],
                  "success": bool("B" in stages and stop == "finish" and harness.termination == "finish"
                                  and evaluation["success"] and final_stop["status"] == "succeeded" and not log.write_error),
                  "stages": stages, "total_requests": sum(a["total_requests"] for a in stages.values()),
                  "final_sim_s": frame.sim_time_s, "final_base_xy_m": frame.base_xy_yaw[:2].tolist(),
                  "independent_evaluation": evaluation, "final_stop": final_stop}
        write(out/"metrics.json", result)
        print(f"{mode}: ended {stop}, A handoff={'B' in stages}, task pass={result['success']}", flush=True)
    finally:
        try:
            if recorder:
                recorder.finish()
            with (out/"trajectory.jsonl").open("w") as stream:
                for row in backend.trajectory:
                    stream.write(json_text(row)+"\n")
        finally:
            if planner:
                planner.close()
            if reader:
                reader.close()
            backend.close()
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor-size-m", nargs=2, type=float, metavar=("LENGTH", "WIDTH"),
                        help="Full platform dimensions along world X and Y in meters (default: task dimensions).")
    parser.add_argument("--max-requests", type=int, default=128)
    parser.add_argument("--max-native-actions", type=int, default=ACTION_LIMIT)
    parser.add_argument("--max-sim-seconds", type=int, default=int(SIM_LIMIT))
    parser.add_argument("--max-wall-seconds", type=int, default=int(WALL_LIMIT))
    parser.add_argument("--max-consecutive-errors", type=int, default=6)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--runs-dir", type=Path, required=True)
    cfg = parser.parse_args()
    if min(cfg.max_requests, cfg.max_consecutive_errors, cfg.max_native_actions, cfg.max_sim_seconds, cfg.max_wall_seconds) < 1:
        parser.error("stage budgets and consecutive error limit must be positive")
    if cfg.mode == "all":
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(run_episode, cfg, mode) for mode in MODES]
            for job in jobs:
                job.result()
    else:
        run_episode(cfg, cfg.mode)
