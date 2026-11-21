"""Fresh, three-arm return ablation from one verified, physically replayed outbound leg."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import time

from .backend import MujocoGo2Backend
from .memory_perception import TaskLabelRGBD, task_targets
from .models import ModelError, _usage_counters
from .occluded_handoff import restore_at_a
from .occluded_return import TASK_ID, fingerprint, write
from .occluded_video import OccludedRecording
from .paired_return_memory import PairedReturnMemory
from .paired_return_planning import PairedReturnPlanner
from .return_route_control import _End, _Follower
from .return_route_tools import RouteModel
from .runner import RunLog, image_data_url, json_text
from .scenarios import get_scenario
from .short_step_clearance import check_clearance


MODES = ("empty", "ledger", "query")
API_LIMIT, ACTION_LIMIT = 32, 115
SIM_LIMIT, WALL_LIMIT, API_TIMEOUT = 470., 1117., 90.


class ReturnFollower(_Follower):
    def __init__(self, harness, log):
        super().__init__(harness, ACTION_LIMIT, lambda action: log.write("native_action", detail=action))
        self.log = log
        self.block = None
        self.refusals = 0

    def act(self, name, **args):
        self.check_state()
        if name == "move_relative":
            self.block = check_clearance(self.harness.frame, self.harness.observation(), **args)
            if self.block is not None:
                self.refusals += 1
                self.log.write("clearance_refusal", arguments=args, public_feedback=self.block)
                raise _End("replan", "current_depth_blocks_requested_translation")
        return super().act(name, **args)

    def execute(self, name, args, turn):
        self.block = None
        self.phase = f"return-{turn}"
        start, completed = len(self.actions), len(self.completed)
        status, reason = "ok", None
        try:
            if name == "scan":
                self.act("turn", yaw_rad=args["yaw_rad"])
            else:
                pts = args["waypoints"]
                for i, pt in enumerate(pts):
                    self.go_to({"id": f"{turn}-{i}", **pt})
                if args["after"] == "dock":
                    self.dock(pts[-1].get("yaw"))
                    status = "finish"
                elif "yaw" in pts[-1]:
                    self.face(pts[-1]["yaw"], self.params.last_view_face_tolerance_rad)
        except _End as exc:
            status, reason = exc.reason, exc.detail
            if status == "control_error":
                status = "replan"
        return {"status": status, "error_code": reason, "native_actions": len(self.actions)-start,
                "completed_waypoints": len(self.completed)-completed, "depth_feedback": self.block,
                "current_odometry": self.harness.observation()["local_odometry"]}


def run_arm(args, mode):
    out = args.runs_dir / mode
    out.mkdir(parents=True, exist_ok=False)
    expected = json.loads(args.fingerprint.read_text())
    backend = MujocoGo2Backend(args.assets_dir, seed=0, scenario=get_scenario(TASK_ID), stop_on_first_collision=True)
    log = RunLog(out)
    recorder = planner = reader = None
    try:
        recorder = OccludedRecording(backend, out / "video", return_label=f"{mode.upper()} | MODEL + SHARED CONTROLLER")
        harness, docs, audit = restore_at_a(backend, args.archive)
        if fingerprint(harness, docs, audit) != expected:
            raise ValueError("Frozen A handoff differs from the reference")
        recorder.mark_handoff()
        write(out/"handoff.json", expected)
        write(out/"manifest.json", {"task_id": TASK_ID, "seed": 0, "mode": mode, "model": "deepseek-flash",
              "outbound": "verified replay", "first_contact_terminates": True, "outbound_decisions": audit["source_decisions"],
              "api_limit_including_reader": API_LIMIT, "native_budget": ACTION_LIMIT, "sim_seconds": SIM_LIMIT,
              "wall_seconds": WALL_LIMIT, "api_timeout": API_TIMEOUT, "max_output_tokens": 8192, "thinking": False,
              "return_history": 6, "depth_clearance_shared": True})
        print(f"{mode}: verified A; fresh return begins", flush=True)
        opts = dict(provider="deepseek", model_name="deepseek-flash", api_key_file=args.api_key_file,
                    timeout_s=API_TIMEOUT, max_output_tokens=8192, trust_env=False, history_turns=0)
        planner = PairedReturnPlanner.from_env(**opts)
        if mode == "query":
            reader = RouteModel.from_env(**opts)
        slot = PairedReturnMemory(mode, docs if mode != "empty" else {}, reader)
        follower = ReturnFollower(harness, log)
        detector = TaskLabelRGBD(task_targets(TASK_ID))
        wall_start = time.monotonic()
        start_sim = harness.frame.sim_time_s
        traj_start = len(backend.trajectory) - 1
        backend.deadline_monotonic = wall_start + WALL_LIMIT
        backend.sim_deadline_s = start_sim + SIM_LIMIT
        history, usage = [], Counter()
        num_plans = num_reader = unknown = errors = 0
        stop, reason = "budget_exhausted", "shared_api_limit"
        while num_plans + num_reader < API_LIMIT:
            remaining = backend.deadline_monotonic-time.monotonic()
            if remaining <= 0 or harness.frame.sim_time_s >= backend.sim_deadline_s:
                reason = "wall_or_sim_limit"
                break
            frame = harness.frame
            obs = harness.observation()
            obs["target_candidates"] = detector.inspect(obs, frame.rgb, frame.depth_m, frame.intrinsics)[0]["target_candidates"]
            log.write("planner_input", index=num_plans, image=log.save_frame(frame, f"return-{num_plans:03}"),
                      memory_card=slot.card, current_odometry=obs["local_odometry"],
                      target_candidates=obs["target_candidates"])
            num_plans += 1
            try:
                call = planner.choose(obs, image_data_url(frame.rgb), slot.card, history,
                                      timeout_s=min(API_TIMEOUT, remaining))
            except ModelError as exc:
                log.write("planner_error", index=num_plans, code=exc.code)
                errors += 1
                if errors >= 2:
                    stop, reason = "model_error", exc.code
                    break
                history.append({"tool": "previous_request", "status": "error", "error_code": exc.code})
                continue
            finally:
                counters = _usage_counters(planner.raw_usage)
                usage.update(counters)
                unknown += int("total_tokens" not in counters)
            errors = 0
            name, arguments = call.name, call.arguments
            log.write("model_choice", index=num_plans, tool=name, arguments=arguments, usage=counters)
            if name == "memory_query":
                memory = slot.handle(arguments["question"], timeout_s=min(API_TIMEOUT, backend.deadline_monotonic-time.monotonic()),
                                     remaining_requests=API_LIMIT-num_plans-num_reader)
                num_reader += memory["requests"]
                unknown += memory["unknown_usage_requests"]
                usage.update(memory["known_usage"])
                log.write("memory_result", detail=memory)
                feedback = {"status": memory["status"], "error_code": memory["error"],
                            "returned_waypoints": len((memory["card"] or {}).get("waypoints", []))}
            elif name == "abort":
                stop, reason = "abort", arguments["reason"]
                break
            else:
                feedback = follower.execute(name, arguments, num_plans)
            history.append({"tool": name, "arguments": arguments, "status": feedback["status"], "feedback": feedback})
            log.write("planner_feedback", index=num_plans, detail=feedback)
            print(f"{mode}: {num_plans}+{num_reader} requests, {name} -> {feedback['status']}, {len(follower.actions)} native actions", flush=True)
            if feedback["status"] in {"finish", "collision", "fall", "budget_exhausted"}:
                stop, reason = feedback["status"], feedback.get("error_code")
                break
        final_stop = backend.stop()
        frame = backend.capture()
        log.save_frame(frame, "final")
        log.write_contacts(backend)
        ev = backend.evaluate()
        if backend.collision_terminated:
            stop = "collision"
        elif backend._fallen_ever:
            stop = "fall"
        elif time.monotonic() > backend.deadline_monotonic or frame.sim_time_s > backend.sim_deadline_s:
            stop, reason = "budget_exhausted", "wall_or_sim_limit"
        path = [item["base_xyz_m"][:2] for item in backend.trajectory[traj_start:]]
        result = {"mode": mode, "success": bool(stop == "finish" and harness.termination == "finish" and ev["success"]
                    and final_stop["status"] == "succeeded" and not log.write_error), "termination": stop,
                  "detail": reason, "independent_evaluation": ev, "final_stop": final_stop,
                  "return_path_m": sum(math.dist(a,b) for a,b in zip(path,path[1:])),
                  "return_sim_seconds": frame.sim_time_s-start_sim, "return_wall_seconds": time.monotonic()-wall_start,
                  "planner_requests": num_plans, "reader_requests": num_reader, "total_requests": num_plans+num_reader,
                  "known_usage": dict(usage), "unknown_usage_requests": unknown, "depth_refusals": follower.refusals,
                  "native_attempts": dict(Counter(a["tool"] for a in follower.actions))}
        write(out/"metrics.json", result)
        print(f"{mode}: {stop}, independent pass={result['success']}", flush=True)
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
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fingerprint", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    cfg = parser.parse_args()
    if cfg.mode == "all":
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(run_arm, cfg, mode) for mode in MODES]
            for job in jobs:
                job.result()
    else:
        run_arm(cfg, cfg.mode)
