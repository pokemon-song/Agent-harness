"""Episode orchestration and an independent, privileged final evaluator."""

import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from io import BytesIO
import json
import math
from pathlib import Path
import time
import uuid

from PIL import Image

from .harness import Harness
from .models import ModelError, build_system_prompt, redact_message
from .provenance import digest, git_identity, runtime_identity, source_hashes
from . import __version__
from .robots.contracts import IndependentTaskEvaluator, SimulationEpisodeBackend


TASK = "找到红色标记，移动到距离标记约 1 米的位置，朝向标记并停止，然后调用 finish。距离按机身参考点到标记的水平距离理解。"


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_text(data) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False, default=_json_default)


def _safe_error(exc: BaseException) -> dict:
    message = redact_message(str(exc))
    return {"type": type(exc).__name__, "code": getattr(exc, "code", "RUNTIME_ERROR"), "message": message[:1000]}


class RunLog:
    def __init__(self, directory: Path):
        self.directory = directory
        (directory / "frames").mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.events = (directory / "events.jsonl").open("w", encoding="utf-8")
        self.write_error = None

    def write(self, event: str, **data):
        line = json_text({"event": event, "wall_elapsed_s": time.monotonic()-self.started, **data}) + "\n"
        try:
            self.events.write(line)
            self.events.flush()
        except OSError as exc:
            # Preserve the resource cleanup path even if the disk becomes full.
            self.write_error = _safe_error(exc)

    def save_frame(self, frame, prefix: str) -> str:
        path = Path("frames") / f"{prefix}-{frame.frame_id}.png"
        Image.fromarray(frame.rgb).save(self.directory / path)
        return path.as_posix()

    def close(self):
        try:
            self.events.close()
        except OSError as exc:
            self.write_error = _safe_error(exc)


def image_data_url(rgb) -> str:
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _task_context(backend, task: str | None = None) -> dict:
    scenario = getattr(backend, "scenario", None)
    return {
        "instruction": task or getattr(backend, "task_instruction", None) or getattr(scenario, "instruction", TASK),
        "task_id": getattr(scenario, "task_id", "red_marker"),
        "source": getattr(scenario, "source", "original_go2_demo"),
        "category": getattr(scenario, "category", "baseline"),
        "goal_kind": getattr(scenario, "goal_kind", "standoff"),
        "standoff_m": getattr(scenario, "standoff_m", 1.0),
        "distance_tolerance_m": getattr(scenario, "distance_tolerance_m", 0.18),
    }


def build_manifest(mode, seed, budgets, backend, model, *, task: str | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    code_hashes = source_hashes(root)
    versions = {}
    for package in ("mujoco", "numpy", "torch", "opencv-python", "httpx", "Pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_recorded"
    context = _task_context(backend, task)
    profile = backend.profile
    posture_supported = profile.capabilities["height_pitch_control"]
    evaluation = backend.evaluator.criteria(standoff_m=context["standoff_m"],
                                            tolerance_m=context["distance_tolerance_m"])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "seed": seed,
        "application_version": __version__, "git": git_identity(root),
        "task": context["instruction"], "task_id": context["task_id"],
        "scenario_source": context["source"], "category": context["category"],
        "model": getattr(model, "model", "unknown_api_model"),
        **runtime_identity(profile), "backend": type(backend).__name__,
        "odometry_source": profile.odometry_source,
        "posture_control_supported": posture_supported,
        "moving_posture_control_supported": profile.capabilities["moving_height_pitch_control"],
        "task_agent": False, "external_memory": False,
        "budgets": budgets, "package_versions": versions, "code_sha256": code_hashes,
        "api_settings": {"timeout_s": getattr(model, "timeout_s", None),
                         "max_output_tokens": getattr(model, "max_output_tokens", 8192),
                         "thinking": getattr(model, "thinking_mode",
                                             getattr(model, "extra_body", {}).get("thinking")),
                         "wire_protocol": getattr(model, "wire_protocol", None),
                         "token_budget_field": getattr(model, "token_budget_field", None),
                         "wire_parameters": {k: v for k, v in getattr(model, "extra_body", {}).items()
                                             if k in {"thinking", "enable_thinking", "preserve_thinking"}},
                         "endpoint": getattr(model, "endpoint", None),
                         "reasoning_history": "complete_retained_assistant_turns_in_memory",
                         "reasoning_logged": False},
        "asset_lock_sha256": hashlib.sha256((root / "assets.lock.json").read_bytes()).hexdigest()
            if (root / "assets.lock.json").exists() else None,
        "evaluation": evaluation,
    }


def run_episode(backend, model, *, mode: str, run_dir: Path, seed: int = 0,
                max_model_calls: int = 30, max_wall_seconds: float = 300,
                max_sim_seconds: float = 120, history_turns: int = 6,
                task: str | None = None, memory_mode: str = "off",
                memory_dir: Path | None = None, task_extension=None) -> dict:
    """Run an evaluated, paused simulation episode and always close owned resources.

    Harness itself accepts the smaller RobotBackend contract. This benchmark runner
    additionally needs a simulation clock, localized RGB observations, trajectory
    evidence and an independent evaluator (SimulationEpisodeBackend). A hardware
    executor with continuous control is deliberately not implemented here.
    """
    if mode != "api":
        raise ValueError("Only API-driven episodes are supported")
    if memory_mode not in {"off", "read_only", "read_write"}:
        raise ValueError("memory_mode must be off, read_only or read_write")
    if task_extension is not None and memory_mode == "off":
        raise ValueError("Task extensions require memory_mode read_only or read_write")
    log = RunLog(run_dir)
    budgets = {"max_model_calls": max_model_calls, "max_wall_seconds": max_wall_seconds,
               "max_sim_seconds": max_sim_seconds, "history_turns": history_turns}
    termination, error, final_stop = "runtime_error", None, None
    history, usage = [], {}
    calls_made, motion_calls, posture_calls = 0, 0, 0
    reasoning_responses, reasoning_characters = 0, 0
    model_wait_s, action_wall_s = 0.0, 0.0
    harness = None
    memory_agent, memory_store = None, None
    memory_initial = None
    memory_tool_calls, memory_io_wall_s = 0, 0.0
    task_tool_calls = 0
    deadline = time.monotonic() + max_wall_seconds
    start_sim = None
    metrics = {}
    context = _task_context(backend, task)
    try:
        backend.deadline_monotonic = deadline
        if (not isinstance(backend, SimulationEpisodeBackend)
                or not isinstance(backend.evaluator, IndependentTaskEvaluator)
                or backend.profile.simulation_mode != "paused_during_model_inference"):
            raise ValueError("This runner requires a paused SimulationEpisodeBackend with an independent evaluator")
        if not backend.profile.capabilities["rgb"] or not backend.profile.capabilities["odometry"]:
            raise ValueError("This simulation benchmark requires RGB and odometry capabilities")
        harness = Harness(backend, context["instruction"])
        frame = harness.frame
        if (frame.sim_time_s is None or not math.isfinite(frame.sim_time_s)
                or frame.base_xy_yaw is None or len(frame.base_xy_yaw) != 3
                or not all(math.isfinite(float(value)) for value in frame.base_xy_yaw)):
            raise ValueError("Simulation benchmark requires a finite simulation timestamp and measured base pose")
        model.system_prompt = build_system_prompt(backend.profile)
        if memory_mode != "off":
            from .agent import MemoryAgent
            from .memory import MemoryScope, MemoryStore
            identity = runtime_identity(backend.profile)
            scope = MemoryScope(backend.profile.robot_id, backend.profile.controller_id,
                                identity["robot_profile_sha256"])
            directory = Path(memory_dir) if memory_dir is not None else Path(__file__).resolve().parents[1] / "memory"
            episode_id = uuid.uuid4().hex
            memory_store = MemoryStore(directory / "memory.sqlite3", scope=scope,
                                       mode=memory_mode, episode_id=episode_id)
            memory_agent = MemoryAgent(harness, memory_store, episode_id, task_extension=task_extension)
            memory_initial = memory_agent.snapshot()
            (run_dir / "memory-initial.json").write_text(json_text(memory_initial), encoding="utf-8")
            model.system_prompt = memory_agent.system_prompt
        tools = memory_agent.tools if memory_agent is not None else harness.tools
        manifest = build_manifest(mode, seed, budgets, backend, model, task=context["instruction"])
        if memory_agent is not None:
            manifest.update(task_agent=True, external_memory=True, experiment_kind="memory-demo-v1",
                            memory_mode=memory_mode, memory_episode_id=episode_id,
                            memory_initial_sha256=memory_initial["sha256"],
                            memory_info=memory_store.info())
            if task_extension is not None:
                manifest["task_extension_id"] = task_extension.extension_id
        manifest.update(system_prompt_sha256=hashlib.sha256(model.system_prompt.encode()).hexdigest(),
                        tools_sha256=digest(tools))
        (run_dir / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
        (run_dir / "tools.json").write_text(json_text(tools), encoding="utf-8")
        (run_dir / "system_prompt.txt").write_text(model.system_prompt, encoding="utf-8")
        (run_dir / "robot_profile.json").write_text(json_text(backend.profile.to_dict()), encoding="utf-8")
        start_sim = harness.frame.sim_time_s
        # Privileged scoring anchor, archived separately from model observations.
        # The physics trace starts after the first control step; keep its exact origin.
        log.write("episode_start", sim_time_s=start_sim,
                  base_xy_m=[float(v) for v in harness.frame.base_xy_yaw[:2]],
                  yaw_rad=float(harness.frame.base_xy_yaw[2]))
        backend.sim_deadline_s = start_sim + max_sim_seconds
        for index in range(max_model_calls):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or harness.frame.sim_time_s-start_sim >= max_sim_seconds:
                termination = "budget_exhausted"
                break
            observation = (memory_agent.observation(remaining_wall_s=remaining,
                                                    remaining_calls=max_model_calls - calls_made)
                           if memory_agent is not None else harness.observation())
            frame_path = log.save_frame(harness.frame, f"{index:03d}")
            log.write("model_input", turn=index, observation=observation, image_path=frame_path,
                      history_turns=min(len(history), history_turns))
            calls_made += 1
            started = time.monotonic()
            try:
                decision = model.decide(observation, image_data_url(harness.frame.rgb), tools,
                                        history, timeout_s=remaining)
            finally:
                model_wait_s += time.monotonic() - started
            reasoning = getattr(decision, "reasoning_content", None)
            reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
            reasoning_responses += int(reasoning_chars > 0)
            reasoning_characters += reasoning_chars
            calls = [asdict(call) for call in decision.calls]
            log.write("model_response", turn=index, calls=calls, usage=decision.usage,
                      model=decision.model, latency_s=time.monotonic()-started,
                      reasoning_present=reasoning_chars > 0, reasoning_chars=reasoning_chars)
            for key, value in decision.usage.items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
            if time.monotonic() >= deadline:
                termination = "budget_exhausted"
                log.write("discarded_response", reason="wall_deadline_exceeded")
                break
            action_started = time.monotonic()
            task_names = memory_agent.task_tool_names if memory_agent is not None else frozenset()
            memory_turn = (memory_agent is not None
                           and any(call.name.startswith("memory_") and call.name not in task_names
                                   for call in decision.calls))
            try:
                results = (memory_agent.dispatch_many(decision.calls) if memory_agent is not None
                           else harness.dispatch_many(decision.calls))
            finally:
                elapsed = time.monotonic() - action_started
                if memory_turn:
                    memory_io_wall_s += elapsed
                else:
                    action_wall_s += elapsed
            memory_tool_calls += sum(call.name.startswith("memory_") and call.name not in task_names
                                     for call in decision.calls) if memory_agent else 0
            task_tool_calls += sum(call.name in task_names for call in decision.calls)
            evidence_ref = (memory_agent.record_public_turn(observation, decision.calls, results)
                            if memory_agent is not None else None)
            motion_calls += sum(c.name in {"move_relative", "turn"} and not r.get("deduplicated")
                                and r.get("error_code") not in {"INVALID_ARGUMENTS", "ONE_TOOL_PER_TURN", "CALL_ID_REUSED",
                                                                "UNSUPPORTED_CAPABILITY", "EPISODE_ENDED"}
                                for c, r in zip(decision.calls, results))
            posture_calls += sum(c.name in {"set_posture", "reset_posture"} and not r.get("deduplicated")
                                 and r.get("error_code") not in {"INVALID_ARGUMENTS", "ONE_TOOL_PER_TURN", "CALL_ID_REUSED",
                                                                 "UNSUPPORTED_CAPABILITY", "EPISODE_ENDED"}
                                 for c, r in zip(decision.calls, results))
            history.append({"observation": observation, "calls": calls, "results": results,
                            "reasoning_content": reasoning,
                            "assistant_content": getattr(decision, "assistant_content", None)})
            history = history[-history_turns:] if history_turns else []
            log.write("tool_results", turn=index, results=results,
                      **({"public_evidence_ref": evidence_ref} if memory_agent is not None else {}),
                      next_observation=harness.observation())
            print(f"[{index+1:02d}] " + ", ".join(f"{r['tool']}: {r['status']}" for r in results), flush=True)
            if time.monotonic() >= deadline or harness.frame.sim_time_s-start_sim >= max_sim_seconds:
                termination = "budget_exhausted"
                break
            if harness.termination is not None:
                termination = harness.termination
                break
            # A fall/collision ends the run; these are proprioception/execution events, not goal truth.
            if any(r.get("error_code") in {"ROBOT_FELL", "COLLISION"} for r in results):
                termination = "execution_failure"
                break
        else:
            termination = "budget_exhausted"
    except KeyboardInterrupt:
        termination = "interrupted"
    except ModelError as exc:
        termination, error = "model_error", _safe_error(exc)
    except Exception as exc:
        termination, error = "runtime_error", _safe_error(exc)
    finally:
        # A simulator continues stepping under zero velocity for a bounded settling interval.
        # Do not equate successful submission of zero velocity with confirmed stopping.
        try:
            action_started = time.monotonic()
            final_stop = backend.stop()
            log.write("final_stop", result=final_stop)
        except BaseException as exc:
            final_stop = {"status": "failed", "error_code": "STOP_EXCEPTION", "detail": _safe_error(exc)}
            log.write("final_stop", result=final_stop)
        finally:
            action_wall_s += time.monotonic() - action_started
        try:
            final_frame = backend.capture()
            log.save_frame(final_frame, "final")
            evaluation = backend.evaluator.evaluate(standoff_m=context["standoff_m"],
                                                    tolerance_m=context["distance_tolerance_m"])
            if task_extension is not None and memory_agent is not None:
                evaluation = dict(evaluation)
                physical_success = evaluation["success"]
                if type(physical_success) is not bool:
                    raise TypeError("Physical evaluator success must be a native bool")
                evaluation["physical_success"] = physical_success
                try:
                    extension_evaluation = task_extension.evaluate()
                    if (not isinstance(extension_evaluation, dict)
                            or type(extension_evaluation.get("success")) is not bool):
                        raise TypeError("Task extension evaluator must return a dictionary with native bool success")
                    json_text(extension_evaluation)
                except Exception as exc:
                    error = _safe_error(exc)
                    termination = "evaluation_error"
                    extension_evaluation = {"success": False, "error": error}
                evaluation.update(task_extension=extension_evaluation,
                                  success=physical_success and extension_evaluation["success"])
            sim_elapsed = final_frame.sim_time_s - (start_sim if start_sim is not None else 0)
            if termination == "finish" and (sim_elapsed > max_sim_seconds or time.monotonic() > deadline):
                termination = "budget_exhausted"
            if log.write_error is not None:
                termination, error = "logging_error", log.write_error
            metrics = {
                "mode": mode, "seed": seed, "termination": termination,
                "task_id": context["task_id"], "scenario_source": context["source"],
                "success": bool(termination == "finish" and evaluation["success"]
                                and final_stop.get("status") == "succeeded"
                                and sim_elapsed <= max_sim_seconds
                                and time.monotonic() <= deadline),
                "error": error, "final_stop": final_stop,
                "model_calls": calls_made, "motion_calls": motion_calls,
                "posture_calls": posture_calls, "usage": usage,
                "reasoning_responses": reasoning_responses, "reasoning_characters": reasoning_characters,
                "wall_elapsed_s": time.monotonic()-log.started, "sim_elapsed_s": sim_elapsed,
                "model_wait_s": model_wait_s, "action_execution_wall_s": action_wall_s,
                "evaluation": evaluation,
                "simulation_mode": backend.profile.simulation_mode,
                "robot_id": backend.profile.robot_id,
                "controller_id": backend.profile.controller_id,
                "robot_profile_sha256": runtime_identity(backend.profile)["robot_profile_sha256"],
                "is_llm_result": mode == "api",
            }
            if memory_mode != "off":
                metrics.update(external_memory=True, experiment_kind="memory-demo-v1", memory_mode=memory_mode,
                               memory_tool_calls=memory_tool_calls, memory_io_wall_s=memory_io_wall_s,
                               memory_initial_sha256=memory_initial["sha256"] if memory_initial else None)
            if task_extension is not None:
                metrics.update(task_extension_id=task_extension.extension_id, task_tool_calls=task_tool_calls)
            if memory_agent is not None:
                try:
                    final_memory = memory_agent.snapshot()
                    (run_dir / "memory-final.json").write_text(json_text(final_memory), encoding="utf-8")
                    metrics.update(memory_final_sha256=final_memory["sha256"], memory_info=memory_store.info())
                except Exception as exc:
                    metrics["memory_archive_error"] = _safe_error(exc)
            (run_dir / "metrics.json").write_text(json_text(metrics), encoding="utf-8")
            with (run_dir / "trajectory.jsonl").open("w", encoding="utf-8") as stream:
                for row in getattr(backend, "trajectory", []):
                    stream.write(json_text(row) + "\n")
            log.write("episode_end", metrics=metrics)
        except BaseException as exc:
            metrics = {"success": False, "termination": "evaluation_error", "original_termination": termination,
                       "task_id": context["task_id"], "scenario_source": context["source"],
                       "error": _safe_error(exc), "final_stop": final_stop, "is_llm_result": mode == "api"}
            if memory_mode != "off":
                metrics.update(external_memory=True, experiment_kind="memory-demo-v1", memory_mode=memory_mode)
            (run_dir / "metrics.json").write_text(json_text(metrics), encoding="utf-8")
            log.write("episode_end", metrics=metrics)
        finally:
            try:
                if hasattr(model, "close"):
                    model.close()
            finally:
                try:
                    backend.close()
                finally:
                    try:
                        if memory_store is not None:
                            memory_store.close()
                    finally:
                        log.close()
    return metrics
