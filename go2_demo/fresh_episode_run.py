"""Run independent live A-to-B episodes. No replay, archive, or checkpoint inputs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256
import math
import multiprocessing
from pathlib import Path
import time
from uuid import uuid4

from .backend import MujocoGo2Backend
from .fresh_exploration import StageExploration
from .fresh_episode_memory import FreshEpisodeMemory, FreshMemoryReader
from .fresh_episode_planning import CONTACT_RULE, FreshEpisodePlanner, public_depth_sectors
from .fresh_subgoal_planning import FreshSubgoalPlanner
from .fresh_subgoals import StageSubgoals
from .harness import Harness
from .memory_perception import TaskLabelRGBD, task_targets
from .models import ModelError, _usage_counters
from .occluded_return import TASK_ID, write
from .occluded_video import OccludedRecording
from .paired_return_run import ACTION_LIMIT, API_TIMEOUT, MODES, SIM_LIMIT, WALL_LIMIT, ReturnFollower
from .perception import PointTracker
from .return_route_control import _End
from .runner import CHECKPOINT_POINTS, RunLog, image_data_url, json_text
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
    def __init__(self, harness, log, memory, stage, *, max_actions=ACTION_LIMIT, actions_per_slice=None):
        super().__init__(harness, log, max_actions=max_actions)
        # A/B share the harness deduplication cache, while each stage starts its
        # own action budget and counter. New stage actions must have fresh IDs.
        self.call_id_prefix = f"stage-{stage}"
        self.target_label = stage
        self.memory = memory
        self.on_action = lambda action: log.write("native_action", stage=stage, detail=action)
        self.watch_for_target = False
        self.actions_per_slice = actions_per_slice
        self.slice_start = 0

    def act(self, name, **args):
        self.check_state()
        if len(self.actions) >= self.max_actions:
            raise _End("budget_exhausted", "native_action_limit")
        if (self.actions_per_slice is not None and
                len(self.actions) - self.slice_start >= self.actions_per_slice):
            raise _End("replan", "execution_slice_complete")
        if self.target_label == "A" and name == "finish":
            raise _End("replan", "A_stop_not_yet_confirmed")
        reply = super().act(name, **args)
        if self.target_label == "A":
            self.memory.observe(self.harness, len(self.actions))
            # Only the milestone bit crosses this boundary. No private geometry goes to the planner.
            if self.harness.backend._ordered_evaluator.a_witness is not None:
                raise _End("handoff", "A_stop_confirmed")
        if self.watch_for_target and name in ("move_relative", "turn"):
            frame = self.harness.frame
            view, _ = self.sensor.inspect(self.harness.observation(), frame.rgb, frame.depth_m, frame.intrinsics)
            if any(c["task_label"] == self.target_label for c in view["target_candidates"]):
                raise _End("replan", "target_visible_in_current_view")
        return reply

    def navigate(self, name, args, turn, target_visible, *, resume_endpoint=False):
        before = self.pose()
        self.slice_start = len(self.actions)
        if name == "follow_waypoints" and args["after"] == "inspect" and not resume_endpoint:
            points = args["waypoints"]
            if all(math.hypot(p["x"]-before[0], p["y"]-before[1]) < self.params.waypoint_tolerance_m for p in points):
                yaw = points[-1].get("yaw", before[2])
                if abs(math.remainder(yaw-before[2], math.tau)) < self.params.last_view_face_tolerance_rad:
                    return {"status": "replan", "error_code": "inspect_waypoint_already_reached",
                            "native_actions": 0, "measured_displacement_m": 0., "measured_yaw_change_rad": 0.}
        self.watch_for_target = name == "follow_waypoints" and args["after"] == "inspect" and not target_visible
        try:
            result = self.execute(name, args, turn, resume_endpoint=resume_endpoint)
        finally:
            self.watch_for_target = False
        after = self.pose()
        return {**result, "measured_displacement_m": round(math.dist(before[:2], after[:2]), 3),
                "measured_yaw_change_rad": round(math.remainder(after[2]-before[2], math.tau), 3)}


def play_stage(stage, mode, harness, log, memory, planner, cfg, *, episode_context=None):
    use_subgoals = getattr(cfg, "navigation", "subgoal") == "subgoal"
    retain = getattr(cfg, "handoff_context", "reset") == "retain"
    if retain and not use_subgoals:
        raise ValueError("Retained dialogue requires the subgoal planner.")
    context = episode_context if episode_context is not None else {}
    follower = StageFollower(harness, log, memory, stage, max_actions=cfg.max_native_actions,
                             actions_per_slice=6 if use_subgoals else None)
    detector = TaskLabelRGBD(task_targets(TASK_ID))
    backend = harness.backend
    t0, wall_start = harness.frame.sim_time_s, time.monotonic()
    backend.deadline_monotonic = wall_start + cfg.max_wall_seconds
    backend.sim_deadline_s = t0 + cfg.max_sim_seconds
    path_start = tuple(harness.frame.base_xy_yaw[:2])
    traj_start = len(backend.trajectory)
    history = context.get("history", []) if retain else []
    usage, choices = Counter(), Counter()
    exploration = None if use_subgoals else StageExploration()
    prior = context.get("task_state") if retain and stage == "B" else None
    subgoals = StageSubgoals(stage, prior_stage_state=prior) if use_subgoals else None
    last_memory = context.get("last_memory") if retain else None
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
        if stage == "A" or use_subgoals:
            obs["depth_sectors"] = public_depth_sectors(frame)
        if subgoals is not None:
            obs["task_state"] = subgoals.observe(obs)
        else:
            obs["exploration"] = exploration.observe(frame, obs, stage)
        obs["native_actions_remaining"] = cfg.max_native_actions - len(follower.actions)
        obs["memory_access"] = memory.query_access()
        obs["latest_memory_feedback"] = last_memory
        retained = planner.context_stats() if retain else {}
        if first_target is None and any(c["task_label"] == stage for c in obs["target_candidates"]):
            first_target = {"sim_time_s": frame.sim_time_s, "planner_decision": planned+1, "image_id": frame.frame_id}
        planned += 1
        log.write("planner_input", stage=stage, index=planned, image=log.save_frame(frame, f"{stage}-{planned:03d}"),
                  memory_card=memory.card, history_turns=min(sum(e.get("stage", stage) == stage and e["tool"] in {"scan", "explore_viewpoint", "follow_waypoints", "continue_subgoal"}
                                                                for e in history), 6),
                  current_odometry=obs["local_odometry"], current_targets=obs["target_candidates"],
                  depth_sectors=obs.get("depth_sectors"), exploration=obs.get("exploration"), task_state=obs.get("task_state"),
                  native_actions_remaining=obs["native_actions_remaining"], memory_access=obs["memory_access"],
                  latest_memory_feedback=last_memory, handoff_context="retain" if retain else "reset",
                  retained_context_turns=retained.get("retained_turns", 0),
                  retained_context_messages=retained.get("retained_messages", 0),
                  retained_context_images=retained.get("retained_images", 0),
                  context_stats=retained)
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
            history.append({"stage": stage, "tool": "previous_request", "status": "error", "error_code": exc.code,
                            "feedback": str(exc) if exc.code == "MODEL_PROTOCOL_ERROR" else exc.code})
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
            answer = memory.handle(args["question"], intents=args["intents"], route_leg=args.get("route_leg", 0),
                                   sighting_window=args.get("sighting_window", "all"),
                                   timeout_s=min(API_TIMEOUT, backend.deadline_monotonic-time.monotonic()),
                                   remaining_requests=cfg.max_requests-planned-reads)
            reads += answer["requests"]
            unknown_usage += answer["unknown_usage_requests"]
            usage.update(answer["known_usage"])
            log.write("memory_result", stage=stage, detail=answer)
            feedback = {"status": answer["status"], "error_code": answer["error"], "items": answer["items"],
                        "cache_hit": answer["cache_hit"], "returned_waypoints": len((answer["card"] or {}).get("waypoints", []))}
            last_memory = {"question": args["question"], **feedback}
        elif name == "set_subgoal" and subgoals is not None:
            feedback = subgoals.set_goal(args, obs)
            log.write("subgoal_set", stage=stage, index=planned, detail=feedback)
        elif name == "abort":
            stop, detail = "abort", args["reason"]
            if retain:
                planner.finish_turn(call, {"status": "abort", "reason": detail})
            log.write("planner_feedback", stage=stage, index=planned, detail={"status": "abort", "reason": detail})
            break
        else:
            chosen = None
            motion_name, motion_args = name, args
            recovery = False
            resume_endpoint = False
            if subgoals is not None:
                recovery = args.get("route_role") == "recovery"
                motion_args = {key: value for key, value in args.items() if key not in {"intention", "route_role"}}
                if name == "continue_subgoal":
                    route = subgoals.pending_route
                    motion_name = "follow_waypoints"
                    motion_args = {"waypoints": route["waypoints"] or [route["endpoint"]], "after": route["after"]}
                    resume_endpoint = not route["waypoints"] and route["endpoint_pending"]
                elif name == "follow_waypoints":
                    subgoals.note_route(motion_args, recovery=recovery)
            elif name == "explore_viewpoint":
                chosen = obs["exploration"]["viewpoints"][args["index"]-1]["waypoint"]
                motion_name, motion_args = "follow_waypoints", {"waypoints": [chosen], "after": "inspect"}
            target_visible = any(c["task_label"] == stage for c in obs["target_candidates"])
            feedback = follower.navigate(motion_name, motion_args, planned, target_visible,
                                         resume_endpoint=resume_endpoint)
            if chosen is not None:
                feedback["chosen_waypoint"] = chosen
                if feedback["status"] == "replan" and feedback["error_code"] != "target_visible_in_current_view":
                    exploration.mark_blocked(chosen)
            if subgoals is not None:
                after, latest = harness.observation(), harness.frame
                after["target_candidates"] = detector.inspect(after, latest.rgb, latest.depth_m, latest.intrinsics)[0]["target_candidates"]
                state = subgoals.record_action(motion_name, motion_args, obs, after, feedback,
                                               intention=args["intention"], recovery=recovery)
                log.write("subgoal_verification", stage=stage, index=planned, task_state=state)
                feedback["subgoal"] = {"id": state["active"]["id"], "status": state["active"]["status"],
                                       "recovery_required": state["recovery_required"]}
            if feedback["measured_displacement_m"] >= .08 or abs(feedback["measured_yaw_change_rad"]) >= .12:
                memory.after_physical_action()
        if retain:
            public_result = {**feedback, "memory_card": memory.card} if name == "memory_query" else feedback
            planner.finish_turn(call, public_result)
        history.append({"stage": stage, "tool": name, "arguments": args, "status": feedback["status"], "feedback": feedback})
        log.write("planner_feedback", stage=stage, index=planned, detail=feedback)
        if planned % 10 == 0 or name in {"follow_waypoints", "explore_viewpoint", "continue_subgoal"} or feedback["status"] == "handoff":
            print(f"{mode} {stage}: {planned}+{reads} requests, {name} -> {feedback['status']}, {len(follower.actions)} native", flush=True)
        if feedback["status"] in {"handoff", "finish", "collision", "fall", "budget_exhausted"}:
            stop, detail = feedback["status"], feedback.get("error_code")
            break
    path = [path_start, *[r["base_xyz_m"][:2] for r in backend.trajectory[traj_start:]]]
    if retain:
        context.update(history=history, last_memory=last_memory, task_state=subgoals.snapshot())
    return {"termination": stop, "detail": detail, "stage": stage, "start_sim_s": t0, "end_sim_s": harness.frame.sim_time_s,
            "wall_seconds": time.monotonic()-wall_start, "path_m": sum(math.dist(a,b) for a,b in zip(path,path[1:])),
            "planner_requests": planned, "reader_requests": reads, "total_requests": planned+reads,
            "known_usage": dict(usage), "unknown_usage_requests": unknown_usage, "first_planner_target_candidate": first_target,
            "planner_tools": dict(choices), "native_attempts": dict(Counter(a["tool"] for a in follower.actions)),
            "boundary_refusals": sum(a["reply"].get("error_code") == "BOUNDARY_BLOCKED" for a in follower.actions),
            "depth_refusals": follower.refusals,
            "task_state": subgoals.snapshot() if subgoals is not None else None,
            "context_stats": planner.context_stats() if retain else {"enabled": False}}


def _collision_score(evaluation, success, penalty_points):
    """Collisions reduce the score independently of ordered task success."""
    goal_points = CHECKPOINT_POINTS * (int(evaluation["checkpoints"]["A"]["passed"]) + int(success))
    collision_count = evaluation.get("collision_count", 0)
    penalty = penalty_points * collision_count
    return {"goal_points": goal_points, "collision_count": collision_count,
            "collision_penalty": penalty, "score": max(0, goal_points - penalty)}


def run_episode(cfg, mode):
    navigation = getattr(cfg, "navigation", "subgoal")
    retain = getattr(cfg, "handoff_context", "reset") == "retain"
    if retain and navigation != "subgoal":
        raise ValueError("Retained dialogue requires --navigation subgoal.")
    scene = episode_scene(cfg.floor_size_m)
    out = cfg.runs_dir / mode
    out.mkdir(exist_ok=False, parents=True)
    backend = MujocoGo2Backend(cfg.assets_dir, seed=cfg.seed, scenario=scene,
                               stop_on_first_collision=False)
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
              "max_output_tokens": 8192, "history": "all_public_dialogue" if retain else 6,
              "reset_model_history_at_A": not retain, "handoff_context": "retain" if retain else "reset",
              "context_retention": "full_public_dialogue" if retain else "stage_local_last_six_actions",
              "retain_memory_card_and_cache_at_A": retain,
              "navigation": "persistent_subgoals_v1" if navigation == "subgoal" else "stage_local_observed_viewpoints",
              "verifier": "public_observation_checks" if navigation == "subgoal" else None,
              "native_actions_per_slice": 6 if navigation == "subgoal" else None,
              "memory_queries": "itemized_evidence_gated",
              "history_unit": "complete_public_tool_turns" if retain else "physical_choices",
              "navigation_reset_at_A": not retain, "new_active_target_at_A": True,
              "fresh_driver_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in
                                      [*sorted(Path(__file__).parent.glob("fresh_*.py")),
                                       Path(__file__).with_name("paired_return_run.py"),
                                       Path(__file__).with_name("return_route_control.py"),
                                       Path(__file__).with_name("occluded_handoff.py"),
                                       Path(__file__).with_name("short_step_clearance.py")]},
              "first_contact_terminates": False, "collision_policy": "record_and_penalize",
              "collision_penalty_points": scene.collision_penalty_points,
              "points_per_checkpoint": CHECKPOINT_POINTS, "completion_requirements": backend.public_task_spec})
        options = dict(provider="deepseek", model_name="deepseek-flash", api_key_file=cfg.api_key_file,
                       timeout_s=API_TIMEOUT, max_output_tokens=8192, trust_env=False, history_turns=0)
        planner_class = FreshSubgoalPlanner if navigation == "subgoal" else FreshEpisodePlanner
        if navigation == "subgoal":
            options["retain_context"] = retain
        planner = planner_class.from_env(**options)
        if mode == "query":
            reader = FreshMemoryReader.from_env(**{k: v for k, v in options.items() if k != "retain_context"})
        memory = FreshEpisodeMemory(mode, TASK_ID, reader)
        assert memory.card is None
        memory.observe(harness, 0)
        print(f"{mode}: live A search from reset simulator", flush=True)
        episode_context = {}
        outbound = play_stage("A", mode, harness, log, memory, planner, cfg, episode_context=episode_context)
        stages = {"A": outbound}
        if outbound["termination"] == "handoff":
            recorder.mark_handoff()
            memory.freeze(preserve_context=retain)
            if retain:
                planner.context_handoff("A", "B")
            log.write("A_handoff", episode_id=episode_id, sim_time_s=harness.frame.sim_time_s,
                      current_odometry=harness.observation()["local_odometry"],
                      handoff_context="retain" if retain else "reset",
                      context_stats=planner.context_stats() if retain else {})
            if not retain:
                harness.tracker = PointTracker()
                harness.tracker.update(harness.frame)
                harness.cache.clear()
            stages["B"] = play_stage("B", mode, harness, log, memory, planner, cfg, episode_context=episode_context)
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
        result.update(_collision_score(evaluation, result["success"], scene.collision_penalty_points))
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
    parser.add_argument("--navigation", choices=("subgoal", "viewpoints"), default="subgoal",
                        help="Persistent model-selected subgoals (default), or legacy observed-viewpoint proposals.")
    parser.add_argument("--handoff-context", choices=("reset", "retain"), default="reset",
                        help="Retain all public planner dialogue/images and memory state across A/B, or reset at A (default).")
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
    if cfg.handoff_context == "retain" and cfg.navigation != "subgoal":
        parser.error("--handoff-context retain requires --navigation subgoal")
    if min(cfg.max_requests, cfg.max_consecutive_errors, cfg.max_native_actions, cfg.max_sim_seconds, cfg.max_wall_seconds) < 1:
        parser.error("stage budgets and consecutive error limit must be positive")
    if cfg.mode == "all":
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(run_episode, cfg, mode) for mode in MODES]
            for job in jobs:
                job.result()
    else:
        run_episode(cfg, cfg.mode)
