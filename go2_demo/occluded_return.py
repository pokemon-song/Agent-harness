"""Paired live returns from the same measured A stop in memory_ab_occluded_v1."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from .backend import MujocoGo2Backend
from .harness import TOOLS
from .memory_perception import TaskLabelRGBD, task_targets
from .models import APIModel, ModelError
from .occluded_handoff import restore_at_a
from .occluded_recall import Recall, RecallResponder
from .occluded_video import OccludedRecording, render
from .runner import RunLog, image_data_url, json_text
from .scenarios import get_scenario


TASK_ID = "memory_ab_occluded_v1"
TASK = (
    "去程在蓝球 A 处停稳，现开始返回找绿方块 B。两段共用出发时的里程计坐标系。"
    "在首次运动和完成前，先使用提供的记忆工具取得去程公开记录。当前记忆内容会保留至下次读取或查询。"
    "历史观测只代表当时的位置；结合当前 RGB、跟踪点几何、实测里程计，自主绕开柜子寻找并重新确认 B。"
    "机身距 B 中心 0.82–1.18 米、朝向误差不超过 0.25 rad 时调用 stop，再调用 finish。"
    "首次与场景物体发生碰撞即终止。每次决策只选择一个工具。"
)
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
MOVEMENT = {"move_relative", "turn", "finish"}
HISTORY = 6


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


class ReturnPlanner(APIModel):
    """Record returned provider usage even if a response fails tool parsing."""

    raw_usage = None
    response_finish_reason = None

    async def _request(self, payload, timeout_s):
        self.raw_usage = None
        self.response_finish_reason = None
        body = await super()._request(payload, timeout_s)
        try:
            reply = json.loads(body)
            self.raw_usage = reply.get("usage")
            choices = reply.get("choices")
            if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict):
                reason = choices[0].get("finish_reason")
                if reason in ("length", "tool_calls", "stop", "content_filter"):
                    self.response_finish_reason = reason
        except (ValueError, AttributeError):
            pass
        return body


def fingerprint(harness, docs, audit):
    return {
        "sim_time_s": harness.frame.sim_time_s,
        "first_return_image_sha256": hashlib.sha256(harness.frame.rgb.tobytes()).hexdigest(),
        "first_public_observation_sha256": digest(harness.observation()),
        "outbound_trajectory_sha256": audit["prefix_trajectory_sha256"],
        "documents": {name: hashlib.sha256(body.encode()).hexdigest() for name, body in docs.items()},
    }


def prepare(args):
    shared = args.runs_dir / "shared"
    shared.mkdir(parents=True, exist_ok=False)
    backend = MujocoGo2Backend(args.assets_dir, seed=0, scenario=get_scenario(TASK_ID), stop_on_first_collision=True)
    try:
        harness, docs, audit = restore_at_a(backend, args.archive)
        for name, body in docs.items():
            dest = shared / "documents" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body, encoding="utf-8")
        write(shared / "handoff.private.json", audit)
        write(shared / "fingerprint.json", fingerprint(harness, docs, audit))
    finally:
        backend.close()


def run_return(args, backend, harness, planner, recall, log):
    """Return-only time and request budgets are identical across arms."""
    history = []
    planner_requests = known_responses = decisions = protocol_failures = 0
    tokens, actions = Counter(), Counter()
    deadline = time.monotonic() + args.max_wall_seconds
    start_time = harness.frame.sim_time_s
    backend.deadline_monotonic = deadline
    backend.sim_deadline_s = start_time + args.max_sim_seconds
    detector = TaskLabelRGBD(task_targets(TASK_ID))
    harness.task = TASK
    tools = [*TOOLS, recall.tool]
    stop_reason, error = "budget_exhausted", None
    prior_error = None
    while planner_requests + recall.requests < args.max_requests:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or harness.frame.sim_time_s >= backend.sim_deadline_s:
            break
        frame = harness.frame
        obs = harness.observation()
        if recall.mode == "local-query":
            recall.set_location(obs["local_odometry"], "return_from_A_to_B")
        candidates, _ = detector.inspect(obs, frame.rgb, frame.depth_m, frame.intrinsics)
        obs["target_candidates"] = candidates["target_candidates"]
        obs["return_memory"] = recall.context()
        if prior_error:
            obs["previous_request_error"] = prior_error
        persisted = {key: val for key, val in obs.items() if key != "return_memory"}
        turn = planner_requests
        log.write("model_input", turn=turn, observation=obs, history_turns=len(history),
                  image_path=log.save_frame(frame, f"{turn:03d}"))
        planner_requests += 1
        started = time.monotonic()
        planner.raw_usage = None
        try:
            decision = planner.decide(obs, image_data_url(frame.rgb), tools, history,
                                      timeout_s=min(args.api_timeout, remaining))
            failure = None
        except ModelError as exc:
            decision, failure = None, exc.code
        usage = planner.raw_usage or {}
        for key in TOKEN_KEYS:
            if type(usage.get(key)) is int:
                tokens[key] += usage[key]
        known_responses += int(all(type(usage.get(key)) is int for key in TOKEN_KEYS))
        latency = time.monotonic() - started
        if failure:
            log.write("model_error", turn=turn, error_code=failure, latency_s=latency, usage=usage,
                      finish_reason=getattr(planner, "response_finish_reason", None))
            if not prior_error and failure in {"MODEL_PROTOCOL_ERROR", "MODEL_TIMEOUT", "MODEL_TRANSPORT_ERROR"}:
                prior_error = failure + ": return exactly one valid tool call"
                protocol_failures += 1
                continue
            stop_reason, error = "model_error", failure
            break
        prior_error = None
        decisions += 1
        calls = [asdict(call) for call in decision.calls]
        log.write("model_response", turn=turn, calls=calls, usage=usage, latency_s=latency)
        if time.monotonic() >= deadline:
            break
        results = []
        for call in decision.calls:
            result = {"call_id": call.call_id, "tool": call.name}
            if len(calls) != 1:
                result.update(status="failed", error_code="ONE_TOOL_PER_TURN")
            elif call.name == recall.tool["function"]["name"]:
                left = args.max_requests - planner_requests - recall.requests
                if left <= 0 and recall.needs_request(call.name, call.arguments):
                    result.update(status="failed", error_code="MODEL_REQUEST_BUDGET")
                else:
                    result.update(recall.handle(call.name, call.arguments,
                                  timeout_s=min(args.api_timeout, max(0., deadline - time.monotonic()))))
            elif call.name in MOVEMENT and not recall.read_ready:
                result.update(status="failed", error_code="OUTBOUND_MEMORY_REQUIRED")
            else:
                result = harness.dispatch(call)
            if result.get("status") == "succeeded" and not result.get("deduplicated"):
                actions[call.name] += 1
            results.append(result)
        log.write("tool_results", turn=turn, results=results, sim_time_s=harness.frame.sim_time_s,
                  memory=recall.context() if any(c.name == recall.tool["function"]["name"] for c in decision.calls) else None)
        log.write_contacts(backend, turn=turn)
        history.append({"observation": persisted, "calls": calls, "results": results,
                        "reasoning_content": decision.reasoning_content, "assistant_content": decision.assistant_content})
        history = history[-HISTORY:]
        if backend.collision_terminated:
            stop_reason = "collision"
            break
        if backend._fallen_ever:
            stop_reason = "fall"
            break
        if harness.termination:
            stop_reason = harness.termination
            break
    final_stop = backend.stop()
    frame = backend.capture()
    log.save_frame(frame, "final")
    log.write_contacts(backend)
    evaluation = backend.evaluate()
    elapsed = frame.sim_time_s - start_time
    wall = args.max_wall_seconds - (deadline - time.monotonic())
    if backend.collision_terminated:
        stop_reason = "collision"
    elif backend._fallen_ever:
        stop_reason = "fall"
    elif stop_reason == "finish" and (elapsed > args.max_sim_seconds or wall > args.max_wall_seconds):
        stop_reason = "budget_exhausted"
    memory_stats = recall.stats
    combined = {k: tokens[k] + memory_stats["known_usage"].get(k, 0) for k in TOKEN_KEYS}
    return {
        "success": bool(stop_reason == "finish" and evaluation["success"] and
                        final_stop["status"] == "succeeded" and not log.write_error),
        "termination": stop_reason, "error": error, "evaluation": evaluation, "final_stop": final_stop,
        "return_wall_seconds": wall, "return_sim_seconds": elapsed, "successful_tools": dict(actions),
        "planner": {"requests": planner_requests, "decisions": decisions, "retries_triggered": protocol_failures,
                    "known_usage": dict(tokens), "unknown_usage_requests": planner_requests - known_responses},
        "memory_reader": memory_stats, "total_requests": planner_requests + recall.requests,
        "known_usage": combined, "usage_complete": known_responses == planner_requests and memory_stats["usage_complete"],
    }


def run(args):
    expected = json.loads((args.runs_dir / "shared/fingerprint.json").read_text())
    out = args.runs_dir / args.mode
    out.mkdir(parents=True, exist_ok=False)
    log = RunLog(out)
    backend = recorder = responder = planner = None
    try:
        backend = MujocoGo2Backend(args.assets_dir, seed=0, scenario=get_scenario(TASK_ID), stop_on_first_collision=True)
        recorder = OccludedRecording(backend, out / "video")
        harness, docs, audit = restore_at_a(backend, args.archive)
        actual = fingerprint(harness, docs, audit)
        if actual != expected:
            raise ValueError("Outbound handoff differs from the frozen control")
        recorder.mark_handoff()
        (out / "handoff.private.json").write_text(json_text(audit))
        cfg = {"task": TASK_ID, "return_instruction": TASK, "seed": 0, "model": "deepseek-flash",
               "return_mode": args.mode, "first_contact_terminates": True,
               "max_requests_including_reader": args.max_requests, "max_wall_seconds": args.max_wall_seconds,
               "max_sim_seconds": args.max_sim_seconds, "timeout_s": args.api_timeout, "max_output_tokens": 8192,
               "shared_prefix_source": args.archive.name, "history_turns": HISTORY, "handoff": actual,
               "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob("*.py"))}}
        write(out / "manifest.json", cfg)
        opts = dict(provider="deepseek", api_key_file=args.api_key_file, model_name=cfg["model"],
                    timeout_s=args.api_timeout, max_output_tokens=8192, trust_env=False)
        planner = ReturnPlanner.from_env(**opts, history_turns=HISTORY)
        if args.mode in ("query", "local-query"):
            responder = RecallResponder.from_env(**opts, history_turns=0)
        recall = Recall(docs, args.mode, responder)
        print(f"{args.mode}: verified A handoff; starting live return", flush=True)
        result = run_return(args, backend, harness, planner, recall, log)
        write(out / "metrics.json", result)
        print(f"{args.mode}: {result['termination']}, success={result['success']}, requests={result['total_requests']}", flush=True)
    finally:
        if backend is not None:
            with (out / "trajectory.jsonl").open("w") as stream:
                for row in backend.trajectory:
                    stream.write(json_text(row) + "\n")
        try:
            if recorder:
                recorder.finish()
        finally:
            if planner:
                planner.close()
            if responder:
                responder.close()
            if backend:
                backend.close()
            log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "direct", "query", "local-query", "render"), required=True)
    parser.add_argument("--archive", type=Path, help="Complete archived H2 episode containing the measured A stop")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=115)
    parser.add_argument("--max-wall-seconds", type=float, default=1117.)
    parser.add_argument("--max-sim-seconds", type=float, default=470.)
    parser.add_argument("--api-timeout", type=float, default=90.)
    args = parser.parse_args(argv)
    if args.mode == "render":
        recordings = [args.runs_dir / mode / "video" for mode in ("direct", "query", "local-query")]
        recordings = [p for p in recordings if (p / "capture.json").is_file()]
        if not recordings:
            parser.error("no completed recordings in --runs-dir")
        for directory in recordings:
            if not (directory / "video.json").exists():
                render(directory)
        return 0
    if not args.archive or not args.archive.is_dir():
        parser.error("--archive requires a source episode")
    if any(not math.isfinite(n) or n <= 0 for n in
           (args.max_requests, args.max_wall_seconds, args.max_sim_seconds, args.api_timeout)):
        parser.error("budgets must be finite and positive")
    if args.mode == "prepare":
        prepare(args)
    else:
        run(args)
    return 0
