"""Exercise the Go2 adapter through the public harness using real MuJoCo physics.

No model API or credentials are used. This is an interface/controller smoke test,
not a model evaluation or a task-planning result.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo.harness import Harness
from go2_demo.provenance import runtime_identity, source_hashes
from go2_demo.robots.contracts import SimulationEpisodeBackend
from go2_demo.robots.registry import create_backend, get_profile
from go2_demo.runner import json_text
from go2_demo.scenarios import get_scenario
from go2_demo.types import ToolCall


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="go2")
    parser.add_argument("--backend", default="mujoco")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    profile = get_profile(args.robot, args.backend)
    directory = args.runs_dir / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                                 + "-robot-adapter-validation-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    result = {"kind": "robot_adapter_physics_validation", "is_llm_result": False,
              "api_requests": 0, "model_score": None, "success": False,
              **runtime_identity(profile), "code_sha256": source_hashes(ROOT)}
    backend, events, start_sim = None, [], None
    started = time.monotonic()
    try:
        backend = create_backend(args.robot, args.backend, asset_dir=args.assets_dir,
                                 seed=0, scenario=get_scenario("red_marker"))
        if not isinstance(backend, SimulationEpisodeBackend):
            raise RuntimeError("Adapter does not satisfy the simulation episode contract")
        harness = Harness(backend, backend.task_instruction)
        start_sim = harness.frame.sim_time_s

        def execute(name, arguments=None, *, expected_error=None):
            call = ToolCall(f"validation-{len(events) + 1}", name, arguments or {})
            feedback = harness.dispatch(call)
            events.append({"tool": name, "arguments": call.arguments, "result": feedback})
            if expected_error is not None:
                if feedback.get("status") != "failed" or feedback.get("error_code") != expected_error:
                    raise RuntimeError(f"{name} did not reject with {expected_error}: {feedback}")
            elif feedback.get("status") != "succeeded":
                raise RuntimeError(f"{name} failed: {feedback}")
            return feedback

        execute("observe")
        execute("move_relative", {"forward_m": .3, "lateral_m": 0.0})
        execute("turn", {"yaw_rad": .3})
        execute("stop")
        execute("set_posture", {"body_height_m": .28, "body_pitch_rad": .08})
        execute("move_relative", {"forward_m": .2, "lateral_m": 0.0},
                expected_error="POSTURE_MODE_ACTIVE")
        execute("reset_posture")
        cancel = backend.cancel()
        events.append({"tool": "backend.cancel", "result": cancel})
        if cancel.get("status") != "succeeded":
            raise RuntimeError(f"Idle cancellation did not stop successfully: {cancel}")
        execute("abort", {"reason": "Interface validation complete; no task success claimed."})
        evaluation = backend.evaluator.evaluate()
        if evaluation["collision"] or evaluation["fallen"] or not evaluation["stopped"]:
            raise RuntimeError("Physics validation ended in an unsafe or moving state")
        result.update(success=True, evaluation=evaluation, events=events)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if backend is not None:
            try:
                result["final_stop"] = backend.stop()
                if result["final_stop"].get("status") != "succeeded":
                    result["success"] = False
                result["sim_elapsed_s"] = float(backend.data.time) - (start_sim or 0.0)
                (directory / "trajectory.jsonl").write_text(
                    "".join(json_text(row) + "\n" for row in backend.trajectory), encoding="utf-8")
            except Exception as exc:
                result.update(success=False, cleanup_error={"type": type(exc).__name__, "message": str(exc)})
            finally:
                try:
                    backend.close()
                except Exception as exc:
                    result.update(success=False, close_error={"type": type(exc).__name__, "message": str(exc)})
        result.update(events=events, wall_elapsed_s=time.monotonic() - started)
        (directory / "validation.json").write_text(json_text(result) + "\n", encoding="utf-8")
    print(json.dumps({"success": result["success"], "api_requests": 0, "is_llm_result": False,
                      "record": str(directory / "validation.json"), "error": result.get("error")},
                     ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
