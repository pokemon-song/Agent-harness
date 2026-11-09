"""CLI execution policy: frozen settings, no overwritten trials, no live API calls."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from go2_demo import __main__ as episode_cli
from go2_demo import benchmark_cli as cli


def make_plan(tmp_path, *, name="candidate", provider="compatible", repeats=1, protocol_id=cli.PROTOCOL_ID):
    directory = tmp_path / name
    directory.mkdir()
    plan = cli.create_plan(label=name, model_name="test-vision-model", provider=provider,
                           repeats=repeats, protocol_id=protocol_id)
    path = directory / "plan.json"
    cli.write_json(path, plan)
    return path, plan


@pytest.fixture
def no_api(monkeypatch):
    monkeypatch.setattr(episode_cli, "_make_model", lambda *_args: pytest.fail("Read credentials or created a model"))
    monkeypatch.setattr(episode_cli, "_run_single", lambda *_args: pytest.fail("Unexpected episode execution"))


@pytest.fixture
def recorded_report(monkeypatch):
    """Replace scoring only; scorer/evaluator contracts are tested independently."""
    def report(path):
        plan = json.loads(Path(path).read_text())
        complete = all((Path(path).parent / case["run_dir"] / "metrics.json").is_file()
                       for case in plan["cases"])
        return {"complete": complete, "rankable": complete,
                "score_100": 0 if complete else None, "issues": []}

    monkeypatch.setattr(cli, "score_plan", report)
    return report


def runner(monkeypatch, *, outcomes=None, after_first=None):
    seen = []
    outcomes = iter(outcomes or [])

    def run(arguments, scenario, directory):
        seen.append((arguments, scenario, directory))
        metrics, code = next(outcomes, ({"success": True, "termination": "finish"}, 0))
        cli.write_json(directory / "metrics.json", metrics)
        if len(seen) == 1 and after_first:
            after_first()
        return metrics, code

    monkeypatch.setattr(episode_cli, "_run_single", run)
    return seen


def test_module_entry_point_is_available_without_a_model():
    result = subprocess.run([sys.executable, "-m", "go2_demo.benchmark_cli", "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "plan,run,report,compare" in result.stdout
    assert "five dimensions" in " ".join(result.stdout.split())


def test_plan_freezes_the_default_grid_without_reading_credentials(no_api, tmp_path):
    directory = tmp_path / "new-model"
    assert cli.main(["plan", "--label", "candidate", "--model-name", "test-model",
                     "--provider", "deepseek", "--output-dir", str(directory)]) == 0
    plan = cli.load_plan(directory / "plan.json")
    assert plan["model"] == {"label": "candidate", "requested_model": "test-model", "provider": "deepseek"}
    assert plan["protocol"]["repeat_count"] == 5
    assert len(plan["cases"]) == 65
    assert all(case["seed"] == 0 for case in plan["cases"])
    assert not (directory / "episodes").exists()


def test_qwen_plan_shares_the_frozen_semantic_protocol_with_deepseek(no_api, tmp_path):
    for provider in ("qwen", "deepseek"):
        assert cli.main(["plan", "--label", provider, "--model-name", provider,
                         "--provider", provider, "--output-dir", str(tmp_path / provider)]) == 0
    qwen = cli.load_plan(tmp_path / "qwen" / "plan.json")
    deepseek = cli.load_plan(tmp_path / "deepseek" / "plan.json")
    assert qwen["protocol_sha256"] == deepseek["protocol_sha256"]
    assert qwen["model"]["provider"] == "qwen"


def test_v3_plan_is_explicit_and_includes_five_categories_without_credentials(no_api, tmp_path):
    directory = tmp_path / "composite-protocol"
    assert cli.main(["plan", "--label", "candidate", "--model-name", "test-model",
                     "--provider", "compatible", "--protocol", "go2-model-eval-v3",
                     "--output-dir", str(directory)]) == 0
    plan = cli.load_plan(directory / "plan.json")
    assert plan["protocol"]["id"] == cli.COMPOSITE_PROTOCOL_ID
    assert len(plan["cases"]) == 70
    assert set(plan["protocol"]["group_weights"].values()) == {.2}
    assert not (directory / "episodes").exists()


def test_v3_run_rebuilds_the_selected_protocol_and_executes_its_14_cases(
        monkeypatch, tmp_path, recorded_report):
    path, plan = make_plan(tmp_path, protocol_id=cli.COMPOSITE_PROTOCOL_ID)
    seen = runner(monkeypatch)
    assert cli.main(["run", "--plan", str(path)]) == 0
    assert len(seen) == 14
    assert [scenario.task_id for _, scenario, _ in seen] == [case["task_id"] for case in plan["cases"]]
    assert sum(scenario.goal_kind == "composite" for _, scenario, _ in seen) == 1


def test_legacy_protocol_is_not_a_new_plan_cli_option(no_api, tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(["plan", "--label", "old", "--model-name", "test", "--provider", "compatible",
                  "--protocol", "go2-model-eval-v1", "--output-dir", str(tmp_path / "legacy")])
    assert error.value.code == 2
    assert not (tmp_path / "legacy").exists()


@pytest.mark.parametrize("existing_plan", [False, True])
def test_plan_requires_a_new_directory_and_preserves_existing_content(no_api, tmp_path, existing_plan):
    directory = tmp_path / "existing"
    directory.mkdir()
    destination = directory / "plan.json"
    if existing_plan:
        destination.write_text("existing evidence")
    assert cli.main(["plan", "--label", "candidate", "--model-name", "test-model",
                     "--provider", "compatible", "--output-dir", str(directory)]) == 2
    if existing_plan:
        assert destination.read_text() == "existing evidence"
    else:
        assert not destination.exists()


@pytest.mark.parametrize("arguments", [
    ["--label", ""], ["--model-name", ""], ["--repeats", "0"], ["--repeats", "21"],
])
def test_invalid_plan_identity_or_repeat_count_creates_no_output(no_api, tmp_path, arguments):
    directory = tmp_path / "invalid"
    command = ["plan", "--label", "candidate", "--model-name", "model", "--provider", "compatible",
               "--output-dir", str(directory), *arguments]
    assert cli.main(command) == 2
    assert not directory.exists()


def test_run_uses_the_entire_frozen_grid_and_settings(monkeypatch, tmp_path, recorded_report, capsys):
    path, plan = make_plan(tmp_path)
    key_file = tmp_path / "credentials.txt"
    key_file.write_text("do-not-print-this-secret")
    assets = tmp_path / "external-assets"
    seen = runner(monkeypatch)
    assert cli.main(["run", "--plan", str(path), "--api-key-file", str(key_file),
                     "--assets-dir", str(assets), "--no-env-proxy"]) == 0
    assert [scenario.task_id for _, scenario, _ in seen] == [case["task_id"] for case in plan["cases"]]
    assert len(seen) == 13
    for (args, _, directory), case in zip(seen, plan["cases"]):
        assert directory == path.parent / case["run_dir"]
        assert directory.is_dir()
        assert vars(args) == {
            "robot": "go2", "backend": "mujoco",
            "mode": "api", "provider": "compatible", "model_name": "test-vision-model",
            "api_key_file": key_file, "no_env_proxy": True, "assets_dir": assets,
            "viewer": False, "seed": 0, "max_model_calls": 30, "max_wall_seconds": 300,
            "max_sim_seconds": 120, "history_turns": 6, "api_timeout": 90, "max_output_tokens": 8192,
        }
    assert json.loads((path.parent / "report.json").read_text())["rankable"] is True
    assert "do-not-print-this-secret" not in capsys.readouterr().out


def test_resume_skips_recorded_cases_without_overwriting_any_evidence(monkeypatch, tmp_path, recorded_report):
    path, plan = make_plan(tmp_path)
    first = path.parent / plan["cases"][0]["run_dir"]
    first.mkdir(parents=True)
    (first / "metrics.json").write_text("original metrics")
    (first / "events.jsonl").write_text("original events")
    seen = runner(monkeypatch)
    assert cli.main(["run", "--plan", str(path)]) == 0
    assert len(seen) == len(plan["cases"]) - 1
    assert (first / "metrics.json").read_text() == "original metrics"
    assert (first / "events.jsonl").read_text() == "original events"
    seen.clear()
    assert cli.main(["run", "--plan", str(path)]) == 0
    assert seen == []


def test_existing_incomplete_episode_is_never_replayed(no_api, tmp_path, recorded_report, capsys):
    path, plan = make_plan(tmp_path)
    first = path.parent / plan["cases"][0]["run_dir"]
    first.mkdir(parents=True)
    events = first / "events.jsonl"
    events.write_text("partial original run")
    assert cli.main(["run", "--plan", str(path)]) == 2
    assert events.read_text() == "partial original run"
    assert not (first / "metrics.json").exists()
    assert "Create a new plan" in capsys.readouterr().err
    assert (path.parent / "report.json").exists()


@pytest.mark.parametrize("mutation", ["grid", "hash", "budget", "traversal"])
def test_tampered_plans_never_start_an_episode(no_api, tmp_path, mutation):
    path, plan = make_plan(tmp_path)
    if mutation == "grid":
        plan["cases"].pop()
    elif mutation == "hash":
        plan["protocol_sha256"] = "0" * 64
    elif mutation == "budget":
        plan["protocol"]["budgets"]["max_model_calls"] += 1
    else:
        plan["cases"][0]["run_dir"] = "../escape"
    cli.write_json(path, plan)
    assert cli.main(["run", "--plan", str(path)]) == 2
    assert not (path.parent / "episodes").exists()
    assert not (tmp_path / "escape").exists()


def test_changed_current_environment_cannot_resume_a_frozen_plan(no_api, monkeypatch, tmp_path, recorded_report):
    path, plan = make_plan(tmp_path)
    changed = deepcopy(plan)
    changed["protocol"]["package_versions"]["mujoco"] = "changed"
    monkeypatch.setattr(cli, "create_plan", lambda **_kwargs: changed)
    assert cli.main(["run", "--plan", str(path)]) == 2
    assert not (path.parent / "episodes").exists()


def test_legacy_v1_run_requires_its_original_checkout_or_a_new_plan(
        no_api, monkeypatch, tmp_path, recorded_report, capsys):
    path, plan = make_plan(tmp_path)
    legacy = deepcopy(plan)
    legacy["protocol"]["id"] = "go2-model-eval-v1"
    current = deepcopy(plan)
    current["protocol"]["id"] = "go2-model-eval-v2"
    monkeypatch.setattr(cli, "load_plan", lambda _path: legacy)
    monkeypatch.setattr(cli, "create_plan", lambda **_kwargs: current)
    assert cli.main(["run", "--plan", str(path)]) == 2
    message = capsys.readouterr().err
    assert "original code checkout" in message and "new v2 plan" in message
    assert "offline report and compare" in message
    assert not (path.parent / "episodes").exists()


def test_protocol_is_rechecked_between_new_episodes(monkeypatch, tmp_path, recorded_report):
    path, plan = make_plan(tmp_path)
    changed = deepcopy(plan)
    changed["protocol"]["code_sha256"]["go2_demo/backend.py"] = "changed"
    seen = runner(monkeypatch, after_first=lambda: monkeypatch.setattr(cli, "create_plan", lambda **_kwargs: changed))
    assert cli.main(["run", "--plan", str(path)]) == 2
    assert len(seen) == 1
    assert (seen[0][2] / "metrics.json").is_file()


def test_case_path_cannot_escape_through_an_existing_symlink(no_api, tmp_path, recorded_report):
    path, _ = make_plan(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (path.parent / "episodes").symlink_to(outside, target_is_directory=True)
    assert cli.main(["run", "--plan", str(path)]) == 2
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("termination,code,error,expected", [
    ("configuration_error", 2, None, 2),
    ("interrupted", 130, None, 130),
    ("model_error", 1, {"code": "MODEL_HTTP_ERROR"}, 2),
    ("model_error", 1, {"code": "MODEL_TRANSPORT_ERROR"}, 2),
    ("model_error", 1, {"code": "UNRECOGNIZED_ERROR"}, 2),
    ("model_error", 1, None, 2),
])
def test_configuration_interrupt_and_infrastructure_errors_stop_remaining_calls(
        monkeypatch, tmp_path, recorded_report, termination, code, error, expected):
    path, _ = make_plan(tmp_path)
    seen = runner(monkeypatch, outcomes=[({"success": False, "termination": termination, "error": error}, code)])
    assert cli.main(["run", "--plan", str(path)]) == expected
    assert len(seen) == 1
    assert json.loads((path.parent / "report.json").read_text())["complete"] is False


@pytest.mark.parametrize("termination,error", [
    ("finish", None), ("execution_failure", None), ("budget_exhausted", None),
    ("model_error", {"code": "MODEL_PROTOCOL_ERROR"}),
    ("model_error", {"code": "MODEL_TIMEOUT"}),
])
def test_model_task_failures_are_recorded_without_external_retry_and_do_not_stop_the_grid(
        monkeypatch, tmp_path, recorded_report, termination, error):
    path, plan = make_plan(tmp_path)
    seen = runner(monkeypatch, outcomes=[({"success": False, "termination": termination, "error": error}, 1)])
    assert cli.main(["run", "--plan", str(path)]) == 0
    assert len(seen) == len(plan["cases"])
    assert [directory for _, _, directory in seen].count(seen[0][2]) == 1


def test_interrupt_always_writes_report_and_releases_the_run_lock(monkeypatch, tmp_path, recorded_report):
    path, _ = make_plan(tmp_path)

    def interrupt(*_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(episode_cli, "_run_single", interrupt)
    assert cli.main(["run", "--plan", str(path)]) == 130
    assert (path.parent / "report.json").exists()
    with cli._run_lock(path.parent):
        pass


def test_a_locked_plan_cannot_start_a_second_run(no_api, tmp_path, capsys):
    path, _ = make_plan(tmp_path)
    with cli._run_lock(path.parent):
        assert cli.main(["run", "--plan", str(path)]) == 2
    assert "Another benchmark run" in capsys.readouterr().err
    assert not (path.parent / "report.json").exists()
    assert (path.parent / ".benchmark-run.lock").exists()


@pytest.mark.parametrize("flag", ["--viewer", "--max-model-calls", "--max-wall-seconds", "--seed"])
def test_run_rejects_viewer_and_budget_or_seed_overrides(flag):
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--plan", "plan.json", flag])
    assert error.value.code == 2


def test_report_and_compare_only_recompute_existing_evidence(no_api, monkeypatch, tmp_path, capsys):
    path, _ = make_plan(tmp_path)
    other, _ = make_plan(tmp_path, name="other")
    seen = []
    monkeypatch.setattr(cli, "score_plan", lambda plan: seen.append(("score", plan)) or {"score_100": None, "rankable": False})
    monkeypatch.setattr(cli, "compare_plans", lambda plans: seen.append(("compare", plans)) or {"comparable": False})
    assert cli.main(["report", "--plan", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["rankable"] is False
    destination = tmp_path / "comparison.json"
    assert cli.main(["compare", "--plans", str(path), str(other), "--output", str(destination)]) == 0
    assert json.loads(destination.read_text()) == {"comparable": False}
    assert seen == [("score", path), ("compare", [path, other])]
    assert not (path.parent / "episodes").exists()


def test_legacy_v1_offline_commands_do_not_require_the_current_execution_protocol(
        no_api, monkeypatch, tmp_path, capsys):
    path, _ = make_plan(tmp_path)
    other, _ = make_plan(tmp_path, name="other-legacy")
    calls = []
    legacy_report = {"protocol_id": "go2-model-eval-v1", "score_100": 75., "rankable": True}
    monkeypatch.setattr(cli, "create_plan", lambda **_kwargs: pytest.fail("Offline command rebuilt a current plan"))
    monkeypatch.setattr(cli, "load_plan", lambda _path: pytest.fail("Offline command used execution validation"))
    monkeypatch.setattr(cli, "score_plan", lambda plan: calls.append(("report", plan)) or legacy_report)
    monkeypatch.setattr(cli, "compare_plans", lambda plans: calls.append(("compare", plans)) or {"protocol_id": "go2-model-eval-v1", "rankable": True})
    assert cli.main(["report", "--plan", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == legacy_report
    assert cli.main(["compare", "--plans", str(path), str(other)]) == 0
    assert json.loads(capsys.readouterr().out)["protocol_id"] == "go2-model-eval-v1"
    assert calls == [("report", path), ("compare", [path, other])]


def test_report_output_cannot_overwrite_the_frozen_plan(no_api, monkeypatch, tmp_path):
    path, _ = make_plan(tmp_path)
    original = path.read_bytes()
    monkeypatch.setattr(cli, "score_plan", lambda _path: {"rankable": False})
    assert cli.main(["report", "--plan", str(path), "--output", str(path)]) == 2
    assert path.read_bytes() == original


def test_report_output_cannot_overwrite_episode_evidence(no_api, monkeypatch, tmp_path):
    path, plan = make_plan(tmp_path)
    metrics = path.parent / plan["cases"][0]["run_dir"] / "metrics.json"
    cli.write_json(metrics, {"original": True})
    monkeypatch.setattr(cli, "score_plan", lambda _path: {"rankable": False})
    assert cli.main(["report", "--plan", str(path), "--output", str(metrics)]) == 2
    assert json.loads(metrics.read_text()) == {"original": True}
