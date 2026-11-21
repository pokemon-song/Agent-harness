"""Return from the frozen A handoff with a model-selected route and an odometry follower."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from .backend import MujocoGo2Backend
from .occluded_handoff import restore_at_a
from .occluded_return import TASK, TASK_ID, fingerprint, write
from .occluded_video import OccludedRecording
from .return_route_control import follow_route
from .return_route_tools import RouteModel, plan_return_route
from .runner import RunLog, image_data_url, json_text
from .scenarios import get_scenario


def run(args):
    expected = json.loads((args.runs_dir / "shared/fingerprint.json").read_text())
    out = args.runs_dir / "hybrid"
    out.mkdir(exist_ok=False)
    backend = MujocoGo2Backend(args.assets_dir, seed=0, scenario=get_scenario(TASK_ID), stop_on_first_collision=True)
    log = RunLog(out)
    planner = reader = recorder = None
    try:
        recorder = OccludedRecording(backend, out / "video", return_label="MODEL ROUTE | ODOMETRY CONTROLLER")
        harness, docs, audit = restore_at_a(backend, args.archive)
        if fingerprint(harness, docs, audit) != expected:
            raise ValueError("Outbound handoff differs from the frozen control")
        recorder.mark_handoff()
        write(out / "handoff.json", expected)
        harness.task = TASK
        obs = harness.observation()
        log.save_frame(harness.frame, "handoff")
        wall_start = time.monotonic()
        sim_start = obs["sim_time_s"]
        deadline = wall_start + 1117.
        backend.deadline_monotonic = deadline
        backend.sim_deadline_s = sim_start + 470.
        opts = dict(provider="deepseek", model_name="deepseek-flash", api_key_file=args.api_key_file,
                    timeout_s=90., max_output_tokens=8192, trust_env=False, history_turns=0)
        planner = RouteModel.from_env(**opts)
        reader = RouteModel.from_env(**opts)
        print("Verified A handoff. Requesting a route from the planner and memory reader.", flush=True)
        plan = plan_return_route(planner, reader, obs, image_data_url(harness.frame.rgb), docs,
                                 timeout_s=90., deadline=deadline, max_requests=4,
                                 emit=lambda event: log.write("route_model", detail=event))
        write(out / "plan.json", plan)
        control = None
        if plan["status"] == "follow_route":
            control = follow_route(harness, plan["waypoints"], max_actions=115, max_sim_seconds=470.,
                                   max_wall_seconds=max(0., deadline-time.monotonic()),
                                   on_action=lambda action: log.write("controller_action", detail=action))
            write(out / "control.json", control)
        final_stop = backend.stop()
        frame = backend.capture()
        log.save_frame(frame, "final")
        log.write_contacts(backend)
        evaluation = backend.evaluate()
        termination = control["termination"] if control else plan["status"]
        if backend.collision_terminated:
            termination = "collision"
        elif backend._fallen_ever:
            termination = "fall"
        elif time.monotonic() > deadline or frame.sim_time_s > backend.sim_deadline_s:
            termination = "budget_exhausted"
        result = {
            "success": bool(termination == "finish" and evaluation["success"]
                            and final_stop["status"] == "succeeded" and not log.write_error),
            "termination": termination, "evaluation": evaluation, "final_stop": final_stop,
            "architecture": "DeepSeek planner and memory reader select route; odometry follower and RGB-D docking",
            "model": "deepseek-flash", "seed": 0, "first_contact_terminates": True,
            "api_request_budget": 4, "native_tool_budget": 115, "max_return_sim_seconds": 470.,
            "max_return_wall_seconds": 1117., "api_timeout_seconds": 90., "max_output_tokens": 8192,
            "model_requests": plan["requests"], "model_known_usage": plan["known_usage"],
            "selected_waypoints": plan["waypoints"], "target_evidence": plan["target_evidence"],
            "successful_tools": dict(Counter(a["tool"] for a in control["actions"]
                                             if a["reply"]["status"] == "succeeded")) if control else {},
            "return_sim_seconds": frame.sim_time_s - sim_start, "return_wall_seconds": time.monotonic()-wall_start,
        }
        write(out / "metrics.json", result)
        print(f"Return ended with {termination}, success={result['success']}", flush=True)
    finally:
        try:
            if recorder:
                recorder.finish()
            with (out / "trajectory.jsonl").open("w") as stream:
                for row in backend.trajectory:
                    stream.write(json_text(row) + "\n")
        finally:
            if planner:
                planner.close()
            if reader:
                reader.close()
            backend.close()
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", default=Path("assets"), type=Path)
    parser.add_argument("--api-key-file", type=Path)
    run(parser.parse_args())
