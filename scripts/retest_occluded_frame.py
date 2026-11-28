"""Paired one-step planner test at the archived sixth return observation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from go2_demo.backend import MujocoGo2Backend
from go2_demo.harness import TOOLS
from go2_demo.models import ModelError, SYSTEM_PROMPT
from go2_demo.occluded_frame import restore_frame_six
from go2_demo.occluded_recall import QUERY, Recall, RecallResponder
from go2_demo.occluded_return import ReturnPlanner, TASK_ID, digest, write
from go2_demo.runner import image_data_url
from go2_demo.scenarios import get_scenario
from go2_demo.types import ToolCall


def make_trial_observation(template, public_history, answer):
    obs = deepcopy(template)
    obs["return_memory"]["current"] = deepcopy(answer)
    obs["public_history"] = deepcopy(public_history)
    return obs


def score_action(harness, calls):
    if len(calls) != 1:
        return {"kind": "protocol", "status": "failed", "error_code": "ONE_TOOL_PER_TURN"}
    call = calls[0]
    if call.name == "memory_query":
        question = call.arguments.get("question")
        valid = (set(call.arguments) == {"question"} and isinstance(question, str)
                 and 0 < len(question.strip()) and len(question) <= 400)
        return {"kind": "memory", "status": "requested" if valid else "failed",
                "error_code": None if valid else "INVALID_MEMORY_ARGUMENTS"}
    kinds = {"track_point": "perception", "observe": "perception", "get_geometry": "perception",
             "turn": "motion", "move_relative": "motion", "stop": "stop",
             "finish": "declaration", "abort": "declaration"}
    before = harness.observation()
    result = harness.dispatch(call)
    after = harness.observation()
    p, q = before["local_odometry"], after["local_odometry"]
    return {"kind": kinds.get(call.name, "protocol"), "status": result["status"],
            "error_code": result.get("error_code"), "result": result,
            "translation_m": math.dist(p["translation_m"], q["translation_m"]),
            "yaw_change_rad": math.remainder(q["yaw_rad"] - p["yaw_rad"], 2 * math.pi),
            "elapsed_sim_s": after["sim_time_s"] - before["sim_time_s"],
            "collision": bool(harness.backend.collision_terminated), "fallen": bool(harness.backend._fallen_ever)}


def planner_call(opts, obs, image, tools):
    model = ReturnPlanner.from_env(**opts, history_turns=0)
    started = time.monotonic()
    try:
        decision = model.decide(obs, image, tools, [], timeout_s=opts["timeout_s"])
        calls, error = [asdict(c) for c in decision.calls], None
    except ModelError as exc:
        calls, error = [], exc.code
    finally:
        model.close()
    return {"calls": calls, "error_code": error, "finish_reason": model.response_finish_reason,
            "usage": model.raw_usage, "latency_s": time.monotonic() - started}


def create_world(assets_dir):
    return MujocoGo2Backend(assets_dir, seed=0, scenario=get_scenario(TASK_ID), stop_on_first_collision=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5, choices=range(1, 21))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    opts = dict(provider="deepseek", api_key_file=args.api_key_file, model_name="deepseek-flash",
                timeout_s=90., max_output_tokens=8192, trust_env=False)
    backend = create_world(args.assets_dir)
    try:
        harness, docs, template, history, audit = restore_frame_six(backend, args.archive)
        raw_image = harness.frame.rgb
        Image.fromarray(raw_image).save(args.out / "before.png")
        image = image_data_url(raw_image)
    finally:
        backend.close()
    old = template["return_memory"]["current"]
    responder = RecallResponder.from_env(**opts, history_turns=0)
    try:
        recall = Recall(docs, "local-query", responder)
        recall.set_location(template["local_odometry"], "return_from_A_to_B")
        result = recall.handle("memory_query", {"question": old["question"]}, timeout_s=90.)
        write(args.out / "reader.json", {"outcome": result, "usage": recall.stats})
        if result["status"] != "succeeded":
            raise RuntimeError("The reader did not produce a valid local answer")
        new = recall.context()["current"]
    finally:
        responder.close()
    answers = {"original": old, "local": new}
    write(args.out / "answers.json", answers)
    tools = [*TOOLS, QUERY]
    contexts = {arm: make_trial_observation(template, history, answer) for arm, answer in answers.items()}
    stripped = deepcopy(contexts["local"])
    stripped["return_memory"]["current"] = old
    if stripped != contexts["original"]:
        raise ValueError("A non-answer field differs between arms")
    write(args.out / "manifest.json", {"pairs": args.pairs, "max_output_tokens": 8192, "api_timeout_s": 90,
          "model": opts["model_name"], "per_trial_requests": 1, "planner_retries": 0,
          "history_representation": "identical public observations, actions and results in one user message",
          "immutable_public_input_sha256": digest(make_trial_observation(template, history, None)),
          "system_sha256": digest(SYSTEM_PROMPT), "tools_sha256": digest(tools), "physics": audit})
    pending = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for n in range(1, args.pairs + 1):
            order = ("original", "local") if n % 2 else ("local", "original")
            tasks = {arm: pool.submit(planner_call, opts, contexts[arm], image, tools) for arm in order}
            for arm, future in tasks.items():
                outcome = future.result()
                trial = {"pair": n, "arm": arm, **outcome}
                write(args.out / f"trial-{n}-{arm}.json", trial)
                pending.append(trial)
                print(n, arm, outcome["error_code"] or [c["name"] for c in outcome["calls"]], flush=True)
    for trial in pending:
        if trial["error_code"] is not None:
            trial["execution"] = {"kind": "protocol", "status": "failed", "error_code": trial["error_code"]}
            continue
        backend = create_world(args.assets_dir)
        try:
            harness, _, _, _, again = restore_frame_six(backend, args.archive)
            if again != audit:
                raise ValueError("The state before scoring differs from the verified original frame")
            trial["execution"] = score_action(harness, [ToolCall(**c) for c in trial["calls"]])
            Image.fromarray(harness.frame.rgb).save(args.out / f"after-{trial['pair']}-{trial['arm']}.png")
        finally:
            backend.close()
    write(args.out / "results.json", pending)


if __name__ == "__main__":
    main()
