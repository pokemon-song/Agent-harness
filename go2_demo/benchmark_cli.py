"""Frozen benchmark plans, resumable API runs, and offline score reports."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from .benchmark import (COMPOSITE_PROTOCOL_ID, LEGACY_PROTOCOL_ID, PROTOCOL_ID,
                        compare_plans, create_plan, load_plan, score_plan, write_json)


ROOT = Path(__file__).resolve().parents[1]


def _validate_frozen_plan(plan: dict) -> None:
    """Recreate the protocol from the current source and installed environment."""
    try:
        model = plan["model"]
        protocol_id = plan["protocol"]["id"]
        if protocol_id == LEGACY_PROTOCOL_ID:
            raise ValueError("This legacy protocol requires its original code checkout to run; "
                             "create a new v2 plan or an explicit v3 plan here. Existing v1 results remain available "
                             "through offline report and compare")
        current = create_plan(label=model["label"], model_name=model["requested_model"],
                              provider=model["provider"], repeats=plan["protocol"]["repeat_count"],
                              protocol_id=protocol_id)
        if plan["protocol"]["id"] != current["protocol"]["id"]:
            raise ValueError("This legacy protocol requires its original code checkout to run; "
                             "create a new v2 plan here. Existing v1 results remain available "
                             "through offline report and compare")
        same = (plan["schema_version"] == current["schema_version"]
                and plan["protocol"] == current["protocol"]
                and plan["protocol_sha256"] == current["protocol_sha256"]
                and plan["cases"] == current["cases"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Plan is missing a valid frozen protocol, model, or case list") from exc
    if not same:
        raise ValueError("Frozen protocol or cases differ from the current source/environment; create a new plan")


def _case_directory(plan_path: Path, case: dict) -> Path:
    relative = case["run_dir"]
    if not isinstance(relative, str) or not relative:
        raise ValueError("Case run_dir must be a nonempty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("Case run_dir must stay inside the plan directory")
    root = plan_path.parent.resolve()
    directory = (root / path).resolve()
    if not directory.is_relative_to(root) or directory == root:
        raise ValueError("Case run_dir must stay inside the plan directory")
    return directory


def _episode_arguments(args, plan: dict, case: dict) -> argparse.Namespace:
    protocol = plan["protocol"]
    budget = protocol["budgets"]
    api = protocol["api_settings"]
    return argparse.Namespace(
        mode="api", provider=plan["model"]["provider"], model_name=plan["model"]["requested_model"],
        api_key_file=args.api_key_file, no_env_proxy=args.no_env_proxy,
        assets_dir=args.assets_dir, viewer=False, seed=case["seed"],
        max_model_calls=budget["max_model_calls"], max_wall_seconds=budget["max_wall_seconds"],
        max_sim_seconds=budget["max_sim_seconds"], history_turns=budget["history_turns"],
        api_timeout=api["timeout_s"], max_output_tokens=api["max_output_tokens"],
    )


def _plan_command(args) -> int:
    plan = create_plan(label=args.label, model_name=args.model_name,
                       provider=args.provider, repeats=args.repeats, protocol_id=args.protocol)
    directory = args.output_dir.expanduser()
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "plan.json"
    write_json(destination, plan)
    print(f"Plan: {destination}")
    return 0


@contextmanager
def _run_lock(directory: Path):
    """Keep one process responsible for a plan, including its final report."""
    import fcntl

    with (directory / ".benchmark-run.lock").open("a", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another benchmark run owns this plan; no episodes were started") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run_command(args) -> int:
    path = args.plan.expanduser().resolve(strict=True)
    with _run_lock(path.parent):
        return _execute_plan(args, path)


def _execute_plan(args, path: Path) -> int:
    plan = load_plan(path)
    stop_code = None
    try:
        _validate_frozen_plan(plan)
        # Importing the existing entry point is harmless; model/backend creation
        # happens only in _run_single, after the frozen plan and case are checked.
        from .__main__ import _run_single
        from .scenarios import get_scenario

        for case in plan["cases"]:
            directory = _case_directory(path, case)
            if (directory / "metrics.json").exists():
                print(f"Recorded, skipping: {case['case_id']}")
                continue
            if directory.exists():
                raise ValueError("Episode directory exists without metrics; cannot overwrite it. Create a new plan")
            # A long run must not silently continue under changed code or packages.
            _validate_frozen_plan(plan)
            scenario = get_scenario(case["task_id"])
            if not scenario.supported:
                raise ValueError("Frozen case requires an unsupported capability")
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError as exc:
                raise ValueError("Episode directory was claimed by another run; existing evidence is preserved") from exc
            metrics, code = _run_single(_episode_arguments(args, plan, case), scenario, directory)
            if code == 130 or metrics.get("termination") == "interrupted":
                stop_code = 130
                break
            if metrics.get("termination") == "configuration_error":
                stop_code = 2
                break
            if metrics.get("termination") == "model_error":
                error = metrics.get("error")
                error_code = error.get("code") if isinstance(error, dict) else None
                if error_code not in {"MODEL_PROTOCOL_ERROR", "MODEL_TIMEOUT"}:
                    stop_code = 2
                    break
    except KeyboardInterrupt:
        stop_code = 130
    finally:
        report = score_plan(path)
        destination = path.parent / "report.json"
        write_json(destination, report)
        print(f"Report: {destination}")
    return stop_code if stop_code is not None else (0 if report.get("rankable") is True else 1)


def _emit_report(report: dict, output: Path | None, plans: list[Path]) -> None:
    if output is not None:
        destination = output.expanduser().resolve()
        for path in plans:
            source = path.expanduser().resolve()
            if destination == source or destination.is_relative_to(source.parent / "episodes"):
                raise ValueError("Report output must not overwrite a plan or its episode evidence")
        write_json(destination, report)
        print(f"Report: {destination}")
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Frozen Go2 model comparison with five dimensions: task success, path efficiency, "
        "wall time, simulation time and model calls. Legacy v1 results support offline report/compare."))
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Create a frozen v2 (65-case) or v3 (70-case) five-dimension plan")
    plan.add_argument("--label", required=True)
    plan.add_argument("--model-name", required=True)
    plan.add_argument("--provider", required=True, choices=("deepseek", "compatible", "qwen"))
    plan.add_argument("--protocol", choices=(PROTOCOL_ID, COMPOSITE_PROTOCOL_ID), default=PROTOCOL_ID,
                      help="v2: four categories at 25%%; v3: five categories including composite at 20%%")
    plan.add_argument("--repeats", type=int, default=5)
    plan.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run", help="Run unrecorded cases at the frozen budgets; viewer is disabled")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--api-key-file", type=Path)
    run.add_argument("--assets-dir", type=Path, default=ROOT / "assets")
    run.add_argument("--no-env-proxy", action="store_true")
    report = commands.add_parser("report", help="Recompute supported v1/v2/v3 scores from evidence without a model")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--output", type=Path)
    compare = commands.add_parser("compare", help="Compare plans under the same frozen protocol without a model")
    compare.add_argument("--plans", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            return _plan_command(args)
        if args.command == "run":
            return _run_command(args)
        if args.command == "report":
            _emit_report(score_plan(args.plan), args.output, [args.plan])
        else:
            _emit_report(compare_plans(args.plans), args.output, args.plans)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
