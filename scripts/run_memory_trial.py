"""Two real API/MuJoCo memory demonstrations, separate from model rankings.

The first instruction explicitly requests an evidence-based lesson; the second
requests recall and evaluation of that lesson. The model chooses query, content
and robot actions. This demonstrates tools, not an unbiased memory ablation.
"""

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo.models import APIModel
from go2_demo.provenance import git_identity, source_hashes
from go2_demo.robots.registry import create_backend
from go2_demo.runner import json_text, run_episode, _safe_error
from go2_demo.scenarios import get_scenario


def _cleanup_owned(backend, model):
    errors, operations = [], []
    if backend is not None:
        operations.extend((("backend.stop", backend.stop), ("backend.close", backend.close)))
    if model is not None:
        operations.append(("model.close", model.close))
    for name, operation in operations:
        try:
            operation()
        except (Exception, KeyboardInterrupt) as exc:
            errors.append({"operation": name, **_safe_error(exc)})
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="go_avoid_texture_drawers")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    scenario = get_scenario(args.task)
    if not scenario.supported:
        parser.error("The selected scene is unsupported")
    directory = args.runs_dir / (datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
                                 + "-memory-api-demo-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    memory_dir = directory / "memory"
    config = {"schema_version": 1, "experiment_kind": "memory-demo-v1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "is_llm_result": True,
              "formal_score": None, "robot_id": "go2", "backend_id": "mujoco",
              "provider": "deepseek", "requested_model": "deepseek-flash", "task_id": scenario.task_id,
              "budgets": {"max_model_calls": 30, "max_wall_seconds": 300, "max_sim_seconds": 120},
              "git": git_identity(ROOT), "code_sha256": source_hashes(ROOT),
              "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "memory_prompt_sha256": hashlib.sha256((ROOT / "go2_demo/prompts/memory_agent.txt").read_bytes()).hexdigest(),
              "design": "Explicit remember/recall instructions; not a causal performance comparison",
              "phases": ["remember", "recall"], "retries": 0}
    (directory / "experiment.json").write_text(json_text(config) + "\n", encoding="utf-8")
    phases = [
        ("remember", "read_write", "这是记忆功能演示。请自主完成上述任务，并在finish之前，"
         "用memory_save保存至少一条由本回合公开观测或实测动作反馈支持的可复用经验。"
         "自行选择值得记录的内容和适用条件；只观察到一次就如实注明，不预先声称任务已被评估为成功。"),
        ("recall", "read_only", "这是记忆复用演示。请自主选择查询词，用memory_search检索之前的经验，"
         "用memory_read读取一条相关记录和证据，判断它是否适用于当前观测，再完成上述任务。"
         "历史位置和跟踪点不能当作当前测量。长期记忆库只读，当前工作笔记仍可更新。"),
    ]
    results = []
    for name, memory_mode, extra in phases:
        backend = model = None
        try:
            if source_hashes(ROOT) != config["code_sha256"]:
                raise RuntimeError("Source changed during the demonstration; start a fresh run")
            if hashlib.sha256((ROOT / "go2_demo/prompts/memory_agent.txt").read_bytes()).hexdigest() != config["memory_prompt_sha256"]:
                raise RuntimeError("Agent prompt changed during the demonstration; start a fresh run")
            print(f"Phase: {name} | Memory: {memory_mode} | Logs: {directory / name}", flush=True)
            model = APIModel.from_env(provider="deepseek", api_key_file=args.api_key_file,
                                      model_name="deepseek-flash", trust_env=False)
            backend = create_backend("go2", "mujoco", asset_dir=ROOT / "assets", seed=0, scenario=scenario)
            metrics = run_episode(backend, model, mode="api", run_dir=directory / name,
                                  task=scenario.instruction + "\n" + extra,
                                  memory_mode=memory_mode, memory_dir=memory_dir,
                                  **config["budgets"])
            backend = model = None  # run_episode closed the owned resources
            events = [json.loads(line) for line in (directory / name / "events.jsonl").read_text().splitlines()]
            successful_tools = [result["tool"] for event in events if event["event"] == "tool_results"
                                for result in event["results"] if result.get("status") == "succeeded"]
            results.append({"phase": name, "metrics": metrics, "successful_tools": successful_tools,
                            "record": str(directory / name / "metrics.json")})
            (directory / "summary.json").write_text(json_text({"experiment_kind": "memory-demo-v1",
                "is_llm_result": True, "formal_score": None, "phases": results}) + "\n", encoding="utf-8")
            # Preserve interrupted and infrastructure-failed attempts; no automatic retries.
            if metrics.get("termination") in {"interrupted", "runtime_error", "evaluation_error", "logging_error"}:
                break
            if (metrics.get("termination") == "model_error"
                    and metrics.get("error", {}).get("code") not in {"MODEL_TIMEOUT", "MODEL_PROTOCOL_ERROR"}):
                break
            if name == "remember" and "memory_save" not in successful_tools:
                break
        except (Exception, KeyboardInterrupt) as exc:
            results.append({"phase": name, "error": _safe_error(exc),
                            "termination": "interrupted" if isinstance(exc, KeyboardInterrupt) else "setup_error"})
            break
        finally:
            cleanup_errors = _cleanup_owned(backend, model)
            if cleanup_errors:
                results[-1]["cleanup_errors"] = cleanup_errors
    memory_hashes = ([results[0].get("metrics", {}).get("memory_final_sha256"),
                      results[1].get("metrics", {}).get("memory_initial_sha256"),
                      results[1].get("metrics", {}).get("memory_final_sha256")]
                     if len(results) == 2 else [])
    demonstrated = (len(results) == 2 and all(item.get("metrics", {}).get("success") for item in results)
                    and not any(item.get("cleanup_errors") or "memory_archive_error" in item.get("metrics", {})
                                for item in results)
                    and "memory_save" in results[0].get("successful_tools", [])
                    and {"memory_search", "memory_read"}.issubset(results[1].get("successful_tools", []))
                    and bool(memory_hashes) and all(isinstance(value, str) and value for value in memory_hashes)
                    and len(set(memory_hashes)) == 1)
    summary = {"experiment_kind": "memory-demo-v1", "is_llm_result": True, "formal_score": None,
               "demonstrated": bool(demonstrated), "phases": results,
               "api_requests": sum(item.get("metrics", {}).get("model_calls", 0) for item in results)}
    (directory / "summary.json").write_text(json_text(summary) + "\n", encoding="utf-8")
    print(json_text({"demonstrated": bool(demonstrated), "report": str(directory / "summary.json")}), flush=True)
    return 0 if demonstrated else 1


if __name__ == "__main__":
    raise SystemExit(main())
