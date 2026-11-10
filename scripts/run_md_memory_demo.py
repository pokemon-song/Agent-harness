"""Two independent agent loops exchanging versioned Markdown; optional real Go2/API.

The small, explicitly instructed task tests handoff and revision, not navigation
or visual identity accuracy. Scripted modes are wiring checks, never LLM scores.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from go2_demo import __version__
from go2_demo.harness import Harness
from go2_demo.md_memory.store import MDMemoryStore
from go2_demo.md_memory.worker import MemoryWorker
from go2_demo.md_memory.planner import MDPlanningAgent, execute_plan, check_plan, PlanError
from go2_demo.models import APIModel
from go2_demo.provenance import git_identity, source_hashes, runtime_identity
from go2_demo.robots.registry import create_backend
from go2_demo.runner import image_data_url


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_error(exc):
    return {"type": type(exc).__name__, "code": getattr(exc, "code", "RUNTIME_ERROR")}


def complete_usage_responses(directory):
    """Count individually complete usage records, not merely nonzero totals."""
    count = 0
    for path in [directory / "memory-agent.jsonl", *directory.glob("planner-*.jsonl")]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("event") != "model_response":
                continue
            usage = record.get("usage", {})
            count += all(type(usage.get(k)) is int and usage[k] >= 0
                         for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    return count


def source_identity():
    paths = [Path(__file__), *sorted((ROOT / "go2_demo/prompts").glob("md_*.txt"))]
    return {**source_hashes(ROOT), **{p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in paths}}


def record_input(store, harness, kind, payload, *, critical=False):
    relative = "evidence/" + uuid.uuid4().hex + ".png"
    path = store.root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(harness.frame.rgb).save(path)
    return store.append_event(kind, {**payload, "observation": harness.observation()},
                              image_path=relative, critical=critical)


def _report(directory, summary):
    phases = summary["phases"]
    text = ["# 双 Agent / Markdown 记忆 Demo", "",
            f"演示通过：{summary['demonstrated']}；真实模型：{summary['is_llm_result']}；真实物理仿真：{summary['simulator_used']}。",
            "", "两个独立模型上下文：后台记忆 Agent 编辑 MD，规划 Agent 只读已发布 MD 并提交短动作。",
            "两阶段共用机器人世界；第二阶段只重建规划模型上下文。程序不替模型选择动作或修复记忆答案。",
            "这是显式记忆交接演示，不是导航能力测试或记忆收益消融。", "",
            f"版本：{summary['application_version']}；冻结提交：`{summary['git']['commit']}`。",
            f"模型调用尝试：{summary['model_requests']}；已知 tokens：{summary['known_total_tokens']}；用量完整：{summary['usage_complete']}。",
            "缺失用量不按零费用处理；脚本模型不是 API 结果。", "",
            "| 阶段 | 记忆版本 | 规划动作 | 符合指令 | 实测动作通过 |",
            "|---|---:|---|---|---|"]
    for phase in phases:
        text.append(f"| {phase['phase']} | {phase.get('memory_revision', '—')} | "
                    f"`{json.dumps(phase.get('plan', {}).get('arguments'), ensure_ascii=False)}` | "
                    f"{phase.get('instruction_match', False)} | {phase.get('execution_match', False)} |")
    text.extend(["", f"旧计划门控：`{summary.get('stale_plan_probe')}`。",
                 f"最终待处理事件数：{summary.get('pending_events')}；最终停止：{summary.get('final_stop', {}).get('status')}。",
                 f"最终碰撞/跌倒：{summary.get('physical', {}).get('collision')} / {summary.get('physical', {}).get('fallen')}。",
                 f"墙钟耗时：{summary['wall_elapsed_s']:.2f} 秒；仿真耗时：{summary.get('sim_elapsed_s')} 秒；路径：{summary.get('path_length_m')} 米。",
                 "", "[结构化结果](summary.json) · [冻结配置](experiment.json) · [MD 记忆目录](memory/revisions/)",
                 "", "MD 保存模型输出的语义内容；证据引用不保证语义解释必然正确。",
                 "执行前按已读文档哈希和未处理关键事件检查；不承诺物理动作中途抢占，也不自动重放中断中的物理动作。"])
    if summary.get("error"):
        text.extend(["", "本轮错误：`" + json.dumps(summary["error"], ensure_ascii=False) + "`。失败原样保留。"])
    (directory / "REPORT.zh-CN.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    video = "execution/third_person/third-person-sim.mp4"
    media = (f'<video controls src="{video}"></video><p>1秒视频对应1秒仿真，模型等待已移除；真实状态回放，无场景重置。</p>'
             if (directory / video).exists() else '<p>本轮没有已渲染录像。</p>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MD记忆双Agent Demo</title>'
            '<style>body{max-width:1000px;margin:32px auto;font:17px/1.7 sans-serif}video{width:100%}pre{white-space:pre-wrap}</style>'
            '<h1>双 Agent / MD 记忆 Demo</h1>' + media + '<p><a href="REPORT.zh-CN.md">报告</a> · '
            '<a href="summary.json">原始结果</a></p><pre>' + html.escape("\n".join(text)) + '</pre></html>')
    (directory / "watch.html").write_text(page, encoding="utf-8")


def run_demo(directory, *, mode="scripted", backend_kind="fixture", api_key_file=None,
             api_timeout=90., max_output_tokens=8192, memory_calls=12, planner_calls=8,
             stage_timeout=300., record=True, render=False):
    if mode not in {"scripted", "api"} or backend_kind not in {"fixture", "mujoco"}:
        raise ValueError("Explicit model mode and backend required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    config = {"experiment_kind": "async-md-memory-demo-v1", "application_version": __version__,
              "git": git_identity(ROOT), "code_sha256": source_identity(),
              "created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "backend": backend_kind,
              "is_llm_result": mode == "api", "simulator_used": backend_kind == "mujoco",
              "formal_score": None, "model": "deepseek-flash" if mode == "api" else "scripted-not-llm",
              "api_timeout_s": api_timeout, "max_output_tokens": max_output_tokens,
              "memory_calls": memory_calls, "planner_calls_per_stage": planner_calls,
              "stage_timeout_s": stage_timeout, "history_turns": 2,
              "recording": record and backend_kind == "mujoco", "automatic_retries": 0,
              "scenario": "two short lateral commands, ACTIVE R2 plus ARCHIVED R1, fresh planner contexts"}
    write_json(directory / "experiment.json", config)
    first = secrets.choice([.25, -.25])
    order_id = "order-" + secrets.token_hex(6)
    expected = {"order_id": order_id, "initial_lateral_m": first, "revised_lateral_m": -first}
    write_json(directory / "contract.private.json", expected)
    summary = {**{k: config[k] for k in ("experiment_kind", "application_version", "git", "is_llm_result", "simulator_used", "formal_score")},
               "demonstrated": False, "phases": [], "error": None, "cleanup_errors": [], "planners": []}
    started = time.monotonic()
    backend = worker = recording = store = None
    initial_sim = None

    def model(role):
        if mode == "api":
            return APIModel.from_env(provider="deepseek", api_key_file=api_key_file,
                                     model_name="deepseek-flash", trust_env=False, timeout_s=api_timeout,
                                     history_turns=2, max_output_tokens=max_output_tokens)
        from go2_demo.md_memory.fixtures import FixtureMemoryModel, FixturePlannerModel
        return FixtureMemoryModel() if role == "memory" else FixturePlannerModel()

    try:
        if backend_kind == "mujoco":
            os.environ.setdefault("MUJOCO_GL", "egl")
            backend = create_backend("go2", "mujoco", asset_dir=ROOT / "assets", seed=0)
            backend.deadline_monotonic = time.monotonic() + 3 * stage_timeout + api_timeout
            backend.sim_deadline_s = float(backend.data.time) + 30.
        else:
            from go2_demo.md_memory.fixtures import FixtureBackend
            backend = FixtureBackend()
        task = f"这是短动作交班演示，订单ID={order_id}。从MD记忆恢复当前生效的动作要求，只执行一次短动作。不要猜测方向。"
        harness = Harness(backend, task)
        initial_sim = harness.frame.sim_time_s
        initial_xy = list(harness.frame.base_xy_yaw[:2])
        write_json(directory / "robot_profile.json", runtime_identity(backend.profile))
        if record and backend_kind == "mujoco":
            recording = backend.start_recording(directory / "execution/third_person")
        store = MDMemoryStore(directory / "memory", scope="world-" + order_id)
        worker = MemoryWorker(store, lambda: model("memory"), log_path=directory / "memory-agent.jsonl",
                              max_calls=memory_calls, request_timeout_s=api_timeout, history_turns=2)
        worker.start()
        previous_plan = None
        for index, (phase, lateral) in enumerate((("initial", first), ("revised", -first))):
            if source_identity() != config["code_sha256"]:
                raise RuntimeError("Source changed during the run")
            direction = "向左" if lateral > 0 else "向右"
            old_direction = "向左" if first > 0 else "向右"
            notice = (f"订单{order_id}，[ACTIVE R{index+1}]：下一次短动作沿机身横向{direction}移动0.25米，前向位移为0。"
                      "这是当前用户指令，请及时保存在任务MD记忆中，供独立规划Agent读取；当前相机图像仅提供现场上下文。")
            if index:
                notice += (f"\n随后回放旧公告：[ARCHIVED R1]：{old_direction}移动0.25米。"
                           "此回放不重新生效，当前ACTIVE R2替代R1。")
            memory_started = time.monotonic()
            event = record_input(store, harness, "user_instruction", {"notice": notice}, critical=True)
            if previous_plan:
                try:
                    check_plan(previous_plan, store, harness.profile)
                    summary["stale_plan_probe"] = {"blocked": False}
                    raise RuntimeError("A pending critical revision did not block the prior plan")
                except PlanError as exc:
                    summary["stale_plan_probe"] = {"blocked": True, "code": exc.code}
                    if exc.code not in {"MEMORY_PENDING", "MEMORY_CHANGED"}:
                        raise
            print(f"{phase}: queued event {event['seq']}; memory agent processing", flush=True)
            worker.notify()
            if not worker.wait_until(event["seq"], timeout_s=stage_timeout):
                raise PlanError("MEMORY_NOT_READY", "Memory processing failed or exceeded stage budget")
            snapshot = store.snapshot()
            result = {"phase": phase, "event_seq": event["seq"], "memory_revision": snapshot["revision"],
                      "memory_ready_wait_s": time.monotonic()-memory_started}
            summary["phases"].append(result)
            planner = MDPlanningAgent(store, model("planner"), harness, log_path=directory / f"planner-{phase}.jsonl",
                                      max_calls=planner_calls, request_timeout_s=api_timeout, history_turns=2)
            try:
                planning_started = time.monotonic()
                plan = planner.plan(task, image_data_url(harness.frame.rgb), timeout_s=stage_timeout)
            finally:
                result["planning_wall_s"] = time.monotonic()-planning_started
                summary["planners"].append(dict(planner.stats))
                try:
                    planner.close()
                except Exception as exc:
                    summary["cleanup_errors"].append({"operation": "planner.close", **safe_error(exc)})
            result["plan"] = plan.public()
            execution_started = time.monotonic()
            feedback = execute_plan(plan, store, harness)
            result["execution_wall_s"] = time.monotonic()-execution_started
            result["feedback"] = feedback
            result["instruction_match"] = (plan.action_name == "move_relative"
                and abs(plan.arguments.get("forward_m", 1)) < 1e-6
                and abs(plan.arguments.get("lateral_m", 1)-lateral) < 1e-6)
            measured = feedback.get("measured", {}).get("translation_start_frame_m")
            result["execution_match"] = bool(result["instruction_match"] and feedback.get("status") == "succeeded"
                and isinstance(measured, list) and len(measured) == 2 and measured[1]*lateral > 0
                and abs(measured[1]) >= .05 and math.dist(measured, [0., lateral]) <= backend.profile.motion.position_tolerance_m)
            record_input(store, harness, "execution_feedback", {"plan": plan.public(), "feedback": feedback})
            previous_plan = plan
            write_json(directory / "progress.json", summary)
            print(f"{phase}: memory v{snapshot['revision']}, action={plan.arguments}, measured_match={result['execution_match']}", flush=True)
            if feedback.get("status") != "succeeded":
                raise PlanError("EXECUTION_FAILED", "Short action failed; stop and replan required")
        # Flush real feedback as another independent memory batch; semantic
        # remembering or ignoring that feedback is still the model's decision.
        worker.notify()
        last_seq = store.snapshot()["latest_event_seq"]
        summary["feedback_processed"] = worker.wait_until(last_seq, timeout_s=stage_timeout)
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = safe_error(exc)
    finally:
        if backend is not None:
            try:
                summary["final_stop"] = backend.stop()
                if backend_kind == "mujoco":
                    physical = backend.evaluator.evaluate()
                    summary["physical"] = {k: physical.get(k) for k in ("collision", "fallen", "stopped", "speed_mps")}
                    trajectory = list(backend.trajectory)
                    positions = [initial_xy, *[r["base_xyz_m"][:2] for r in trajectory]]
                    summary["path_length_m"] = sum(math.dist(a, b) for a, b in zip(positions, positions[1:]))
                    summary["sim_elapsed_s"] = float(backend.data.time) - initial_sim
                    write_json(directory / "trajectory.json", {"initial_xy_m": initial_xy, "points": trajectory})
                else:
                    summary["physical"] = {"collision": None, "fallen": None, "stopped": True}
                    summary["sim_elapsed_s"] = None
                    summary["path_length_m"] = None
            except Exception as exc:
                summary["cleanup_errors"].append({"operation": "stop/evaluate", **safe_error(exc)})
        if worker is not None:
            try:
                summary["worker_stopped"] = worker.stop(timeout_s=api_timeout+5)
                if not summary["worker_stopped"]:
                    raise TimeoutError("Memory worker did not exit")
            except Exception as exc:
                summary["cleanup_errors"].append({"operation": "memory_worker.stop", **safe_error(exc)})
            finally:
                summary["memory_agent"] = dict(worker.stats)
        if store is not None:
            try:
                snapshot = store.snapshot()
                write_json(directory / "memory-final.json", snapshot)
                summary["pending_events"] = snapshot["latest_event_seq"] - snapshot["processed_event_seq"]
                summary["memory_revision"] = snapshot["revision"]
            except Exception as exc:
                summary["cleanup_errors"].append({"operation": "snapshot", **safe_error(exc)})
        for name, close in (("recording", recording.finish if recording else None),
                            ("backend", backend.close if backend else None)):
            if close:
                try:
                    close()
                except Exception as exc:
                    summary["cleanup_errors"].append({"operation": name, **safe_error(exc)})
    summary["wall_elapsed_s"] = time.monotonic()-started
    summary["source_unchanged"] = source_identity() == config["code_sha256"]
    stats = [summary.get("memory_agent", {}), *summary["planners"]]
    summary["model_requests"] = sum(s.get("api_requests", 0) for s in stats)
    summary["api_requests"] = summary["model_requests"] if mode == "api" else 0
    summary["known_total_tokens"] = sum(s.get("usage", {}).get("total_tokens", 0) for s in stats)
    summary["complete_usage_responses"] = complete_usage_responses(directory)
    summary["usage_complete"] = (mode == "api" and summary.get("worker_stopped", False) and all(
        s.get("api_requests", 0) == s.get("completed_responses", 0) for s in stats)
        and summary["model_requests"] > 0
        and summary["complete_usage_responses"] == summary["model_requests"])
    summary["demonstrated"] = bool(not summary["error"] and not summary["cleanup_errors"] and summary["source_unchanged"]
        and len(summary["phases"]) == 2 and all(p.get("execution_match") for p in summary["phases"])
        and summary.get("stale_plan_probe", {}).get("code") in {"MEMORY_PENDING", "MEMORY_CHANGED"}
        and summary.get("feedback_processed") and summary.get("pending_events") == 0
        and summary.get("final_stop", {}).get("status") == "succeeded"
        and not summary.get("physical", {}).get("collision") and not summary.get("physical", {}).get("fallen"))
    if render and recording is not None:
        try:
            from scripts import render_memory_mission_video as video
            video.LABELS["md-memory-demo"] = "MD MEMORY DEMO: CONTINUOUS WORLD, TWO PLANNER CONTEXTS"
            manifest = video.render_phase(directory / "execution/third_person", "md-memory-demo", video.ffmpeg_path(), timebases=("sim",))
            write_json(directory / "video-manifest.json", manifest)
        except Exception as exc:
            summary["video_error"] = safe_error(exc)
    write_json(directory / "summary.json", summary)
    _report(directory, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scripted", "api"), default="scripted")
    parser.add_argument("--backend", choices=("fixture", "mujoco"), default="fixture")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--api-timeout", type=float, default=90.)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--memory-calls", type=int, default=12)
    parser.add_argument("--planner-calls", type=int, default=8)
    parser.add_argument("--stage-timeout", type=float, default=300.)
    parser.add_argument("--no-recording", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.api_timeout) or not 1 <= args.api_timeout <= 180
        or not math.isfinite(args.stage_timeout) or not 1 <= args.stage_timeout <= 900
        or not 1 <= args.max_output_tokens <= 16384
        or not 1 <= args.memory_calls <= 30 or not 1 <= args.planner_calls <= 20):
        parser.error("Budgets are outside supported demo limits")
    directory = args.runs_dir / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                                 + "-md-memory-demo-" + uuid.uuid4().hex[:6])
    print(f"Demo artifacts: {directory}", flush=True)
    summary = run_demo(directory, mode=args.mode, backend_kind=args.backend, api_key_file=args.api_key_file,
                       api_timeout=args.api_timeout, max_output_tokens=args.max_output_tokens,
                       memory_calls=args.memory_calls, planner_calls=args.planner_calls,
                       stage_timeout=args.stage_timeout, record=not args.no_recording, render=args.render_video)
    print(json.dumps({"demonstrated": summary["demonstrated"], "api_requests": summary["api_requests"],
                      "report": str(directory / "REPORT.zh-CN.md"), "error": summary["error"]}, ensure_ascii=False), flush=True)
    return 0 if summary["demonstrated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
