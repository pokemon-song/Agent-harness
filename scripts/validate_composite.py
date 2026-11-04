"""Check the compound scene with scripted waypoints and real MuJoCo dynamics.

This uses simulator poses to follow a known route. It makes no model requests
and produces no model score; its output is a scene reachability check only.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from go2_demo.backend import MujocoGo2Backend
from go2_demo.evaluation_metrics import trajectory_metrics
from go2_demo.harness import Harness
from go2_demo.runner import json_text
from go2_demo.scenarios import get_scenario
from go2_demo.types import ToolCall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    directory = args.runs_dir / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                                 + "-composite-scripted-validation-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    scenario = get_scenario("composite_drawers_gate_b")
    result = {
        "kind": "scripted_scene_reachability_validation", "task_id": scenario.task_id,
        "is_llm_result": False, "api_requests": 0, "model_score": None,
        "uses_privileged_simulator_pose": True, "seed": 0,
        "instruction": scenario.instruction, "success": False,
    }
    backend, events = None, []
    try:
        backend = MujocoGo2Backend(args.assets_dir, seed=0, scenario=scenario)
        harness = Harness(backend, scenario.instruction)
        start_time, started = float(backend.data.time), time.monotonic()
        start = {"sim_time_s": start_time, "base_xy_m": harness.frame.base_xy_yaw[:2].tolist(),
                 "yaw_rad": float(harness.frame.base_xy_yaw[2])}
        Image.fromarray(harness.frame.rgb).save(directory / "initial.png")
        # The deliberately supplied route proves feasibility, not agent planning.
        waypoints = [(0.5, .95), (2.65, .95), (2.65, 0.), (scenario.target.xy[0] - 1., 0.)]
        result["scripted_waypoints_xy_m"] = waypoints
        for waypoint in waypoints:
            for _ in range(20):
                x, y, yaw = harness.frame.base_xy_yaw
                dx, dy = waypoint[0] - x, waypoint[1] - y
                distance = math.hypot(dx, dy)
                if distance <= .10:
                    break
                scale = min(1., .55 / distance)
                call = ToolCall(f"script-{len(events) + 1}", "move_relative", {
                    "forward_m": float(scale * (math.cos(yaw) * dx + math.sin(yaw) * dy)),
                    "lateral_m": float(scale * (-math.sin(yaw) * dx + math.cos(yaw) * dy)),
                })
                feedback = harness.dispatch(call)
                events.append({"call": asdict(call), "result": feedback})
                if feedback["status"] != "succeeded":
                    raise RuntimeError("Scripted move failed: " + str(feedback.get("error_code")))
            else:
                raise RuntimeError("Scripted waypoint did not converge")
        for _ in range(5):
            x, y, yaw = harness.frame.base_xy_yaw
            bearing_error = math.atan2(scenario.target.xy[1] - y, scenario.target.xy[0] - x) - yaw
            bearing_error = math.atan2(math.sin(bearing_error), math.cos(bearing_error))
            if abs(bearing_error) <= .08:
                break
            call = ToolCall(f"script-{len(events) + 1}", "turn", {
                "yaw_rad": max(-.5, min(.5, bearing_error)),
            })
            feedback = harness.dispatch(call)
            events.append({"call": asdict(call), "result": feedback})
            if feedback["status"] != "succeeded":
                raise RuntimeError("Scripted final alignment failed: " + str(feedback.get("error_code")))
        else:
            raise RuntimeError("Scripted final heading did not converge")
        call = ToolCall("script-finish", "finish", {})
        feedback = harness.dispatch(call)
        events.append({"call": asdict(call), "result": feedback})
        evaluation = backend.evaluate()
        result.update(evaluation=evaluation, final_stop=feedback,
                      success=evaluation["success"] and feedback["status"] == "succeeded",
                      wall_elapsed_s=time.monotonic() - started,
                      sim_elapsed_s=float(backend.data.time) - start_time,
                      motion_calls=len(events) - 1)
        result["trajectory_measurements"] = trajectory_metrics(
            backend.trajectory, start=start, final_xy=evaluation["base_xy_m"],
            sim_elapsed_s=result["sim_elapsed_s"], reference={
                "goal_kind": "standoff", "target_xy_m": list(scenario.target.xy),
                "standoff_m": scenario.standoff_m,
                "distance_tolerance_m": scenario.distance_tolerance_m,
            })
        Image.fromarray(harness.frame.rgb).save(directory / "final.png")
        (directory / "trajectory.jsonl").write_text(
            "".join(json_text(row) + "\n" for row in backend.trajectory), encoding="utf-8")
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if backend is not None:
            backend.close()
        (directory / "events.jsonl").write_text(
            "".join(json_text(row) + "\n" for row in events), encoding="utf-8")
        (directory / "validation.json").write_text(json_text(result) + "\n", encoding="utf-8")
    print(json_text({"success": result["success"], "is_llm_result": False,
                     "api_requests": 0, "record": str(directory / "validation.json"),
                     **({"error": result["error"]} if "error" in result else {})}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
