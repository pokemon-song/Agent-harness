"""Run a preregistered, serial AB/BA comparison without replacing any trial."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo.benchmark import create_plan, load_plan, score_plan, compare_plans, write_json
from go2_demo.benchmark_cli import _run_lock, _validate_frozen_plan, _case_directory, _episode_arguments
from go2_demo.__main__ import _run_single
from go2_demo.scenarios import get_scenario


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--deepseek-key-file", type=Path, required=True)
    parser.add_argument("--qwen-key-file", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory = args.experiment_dir.resolve()
    configs = {
        "deepseek": {"label": "DeepSeek-V4.1-Flash", "model_name": "deepseek-flash", "provider": "deepseek",
                     "endpoint": "https://api.deepseek.com", "key_file": args.deepseek_key_file},
        "qwen": {"label": "Qwen3.8-Max", "model_name": "qwen3.8-max", "provider": "qwen",
                 "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1", "key_file": args.qwen_key_file},
    }
    plans = {}
    for name, config in configs.items():
        plan_path = directory / name / "plan.json"
        if args.resume:
            plans[name] = load_plan(plan_path)
            if plans[name]["model"] != {"label": config["label"], "requested_model": config["model_name"], "provider": config["provider"]}:
                raise ValueError("Resume model identity differs")
        else:
            plan = create_plan(label=config["label"], model_name=config["model_name"], provider=config["provider"])
            plan_path.parent.mkdir(parents=True, exist_ok=False)
            write_json(plan_path, plan)
            plans[name] = plan
    if len({p["protocol_sha256"] for p in plans.values()}) != 1:
        raise ValueError("The two frozen protocols differ")
    for plan in plans.values():
        _validate_frozen_plan(plan)
    schedule = []
    for index, case in enumerate(plans["deepseek"]["cases"]):
        for name in (("deepseek", "qwen") if index % 2 == 0 else ("qwen", "deepseek")):
            schedule.append({"model": name, "case_id": case["case_id"], "index": index})
    experiment_path = directory / "experiment.json"
    if args.resume:
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        if experiment["schedule"] != schedule:
            raise ValueError("The preregistered schedule differs")
    else:
        experiment = {
            "kind": "real_model_api_comparison", "created_utc": utc(), "started_utc": utc(), "finished_utc": None,
            "protocol_id": plans["deepseek"]["protocol"]["id"],
            "protocol_sha256": plans["deepseek"]["protocol_sha256"],
            "models": {name: {k: v for k, v in config.items() if k != "key_file"} for name, config in configs.items()},
            "schedule_description": "One episode at a time. Same case paired, alternating AB/BA order. No task-result retry.",
            "schedule": schedule, "automatic_episode_retries": 0,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "budget_per_episode": plans["deepseek"]["protocol"]["budgets"],
            "api_settings_semantic": plans["deepseek"]["protocol"]["api_settings"],
            "sources": ["https://api-docs.deepseek.com/quick_start/pricing/", "https://api-docs.deepseek.com/guides/thinking_mode/",
                        "https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions",
                        "https://help.aliyun.com/zh/model-studio/deep-thinking"],
            "status": "running", "stopped_models": {}, "attempted_episodes": 0,
        }
        write_json(experiment_path, experiment)

    def refresh():
        for name in configs:
            path = directory / name / "plan.json"
            report = score_plan(path)
            write_json(path.parent / "report.json", report)
        comparison = compare_plans([directory / name / "plan.json" for name in configs])
        write_json(directory / "comparison.json", comparison)
        write_json(experiment_path, experiment)

    with ExitStack() as stack:
        for name in configs:
            stack.enter_context(_run_lock(directory / name))
        refresh()
        try:
            for number, item in enumerate(schedule, 1):
                name, index = item["model"], item["index"]
                if name in experiment["stopped_models"]:
                    continue
                config, plan = configs[name], plans[name]
                case = plan["cases"][index]
                plan_path = directory / name / "plan.json"
                case_dir = _case_directory(plan_path, case)
                if (case_dir / "metrics.json").exists():
                    continue
                if case_dir.exists():
                    raise ValueError("A partial attempt exists; evidence will not be overwritten")
                _validate_frozen_plan(plan)
                # Make service routing explicit for every episode, including resumes.
                os.environ["MODEL_BASE_URL"] = config["endpoint"]
                cli = argparse.Namespace(api_key_file=config["key_file"], no_env_proxy=True, assets_dir=ROOT / "assets")
                episode_args = _episode_arguments(cli, plan, case)
                case_dir.mkdir(parents=True, exist_ok=False)
                print(f"COMPARISON START {number}/{len(schedule)} {name} {case['case_id']}", flush=True)
                began = time.monotonic()
                metrics, code = _run_single(episode_args, get_scenario(case["task_id"]), case_dir)
                experiment["attempted_episodes"] += 1
                experiment["last_episode"] = {"model": name, "case_id": case["case_id"], "success": metrics["success"],
                                              "termination": metrics["termination"], "finished_utc": utc(),
                                              "elapsed_including_initialization_s": time.monotonic() - began}
                if code == 130 or metrics["termination"] == "interrupted":
                    experiment["status"] = "interrupted"
                    break
                if (metrics["termination"] in {"configuration_error", "startup_error", "runtime_error", "evaluation_error", "logging_error"}
                        or metrics["termination"] == "model_error" and (metrics.get("error") or {}).get("code") not in {"MODEL_PROTOCOL_ERROR", "MODEL_TIMEOUT"}):
                    experiment["stopped_models"][name] = {"case_id": case["case_id"], "termination": metrics["termination"],
                                                           "error_code": (metrics.get("error") or {}).get("code")}
                refresh()
                print(f"COMPARISON END {number}/{len(schedule)} {name} success={metrics['success']} termination={metrics['termination']}", flush=True)
            else:
                experiment["status"] = "completed" if not experiment["stopped_models"] else "infrastructure_blocked"
        except KeyboardInterrupt:
            experiment["status"] = "interrupted"
        except BaseException:
            experiment["status"] = "runner_error"
            raise
        finally:
            experiment["finished_utc"] = utc()
            refresh()
    return 0 if experiment["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
