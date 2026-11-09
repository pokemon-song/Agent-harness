"""The real-demo driver must preserve summaries on archive and cleanup failures."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_memory_trial.py"
    spec = importlib.util.spec_from_file_location("memory_trial_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "source_hashes", lambda root: {"fixture.py": "frozen"})
    monkeypatch.setattr(module, "git_identity", lambda root: {"commit": "fixture", "dirty": False})
    return module


def resources(driver, monkeypatch, *, failures=()):
    models, backends, operations = [], [], []

    def operation(name):
        def perform():
            operations.append(name)
            if name in failures:
                raise RuntimeError(f"simulated {name} failure")
        return perform

    def model_factory(**kwargs):
        model = SimpleNamespace(close=operation("model.close"))
        models.append(model)
        return model

    def backend_factory(*args, **kwargs):
        backend = SimpleNamespace(stop=operation("backend.stop"), close=operation("backend.close"))
        backends.append(backend)
        return backend

    monkeypatch.setattr(driver.APIModel, "from_env", model_factory)
    monkeypatch.setattr(driver, "create_backend", backend_factory)
    return models, backends, operations


def recorded_episode(driver, monkeypatch, *, missing=None, archive_error=False):
    def run(backend, model, *, run_dir, **kwargs):
        run_dir.mkdir(parents=True)
        phase = run_dir.name
        metrics = {"success": True, "termination": "finish", "model_calls": 2,
                   "memory_initial_sha256": "a" * 64, "memory_final_sha256": "a" * 64}
        if missing and phase == missing[0]:
            metrics.pop(missing[1])
        if archive_error and phase == "recall":
            metrics["memory_archive_error"] = {"type": "OSError", "message": "simulated archive failure"}
        tools = ["memory_save", "finish"] if phase == "remember" else ["memory_search", "memory_read", "finish"]
        event = {"event": "tool_results", "results": [{"tool": tool, "status": "succeeded"} for tool in tools]}
        (run_dir / "events.jsonl").write_text(json.dumps(event) + "\n")
        (run_dir / "metrics.json").write_text(json.dumps(metrics))
        # A normally returning run_episode owns and closes both resources.
        backend.close()
        model.close()
        return metrics

    monkeypatch.setattr(driver, "run_episode", run)


def summary_in(directory):
    paths = list(directory.glob("*/summary.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


@pytest.mark.parametrize("missing", [
    ("remember", "memory_final_sha256"),
    ("recall", "memory_initial_sha256"),
    ("recall", "memory_final_sha256"),
])
def test_missing_memory_archive_hash_keeps_successful_task_records_and_final_summary(driver, monkeypatch, tmp_path, missing):
    models, _, operations = resources(driver, monkeypatch)
    recorded_episode(driver, monkeypatch, missing=missing)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 1
    summary = summary_in(tmp_path)
    assert summary["demonstrated"] is False and len(summary["phases"]) == 2
    assert all(item["metrics"]["success"] for item in summary["phases"])
    assert summary["formal_score"] is None and summary["api_requests"] == 4
    assert operations.count("model.close") == len(models) == 2


def test_reported_memory_archive_error_cannot_claim_demonstration_success(driver, monkeypatch, tmp_path):
    resources(driver, monkeypatch)
    recorded_episode(driver, monkeypatch, archive_error=True)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 1
    assert summary_in(tmp_path)["demonstrated"] is False


@pytest.mark.parametrize("failures", [
    ("backend.stop",), ("backend.close",), ("backend.stop", "backend.close"), ("model.close",),
])
def test_cleanup_failures_still_attempt_every_close_and_write_final_summary(driver, monkeypatch, tmp_path, failures):
    models, backends, operations = resources(driver, monkeypatch, failures=failures)

    def interrupted_episode(*args, **kwargs):
        raise RuntimeError("simulated episode failure before returning ownership")

    monkeypatch.setattr(driver, "run_episode", interrupted_episode)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 1
    summary = summary_in(tmp_path)
    assert summary["demonstrated"] is False and len(summary["phases"]) == 1
    assert len(models) == len(backends) == 1  # Never start recall or retry.
    assert operations == ["backend.stop", "backend.close", "model.close"]
    phase = summary["phases"][0]
    assert phase["error"]["message"] == "simulated episode failure before returning ownership"
    assert [error["operation"] for error in phase["cleanup_errors"]] == list(failures)


def test_complete_normal_demo_still_passes_and_preserves_shared_memory_hashes(driver, monkeypatch, tmp_path):
    _, _, operations = resources(driver, monkeypatch)
    recorded_episode(driver, monkeypatch)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 0
    summary = summary_in(tmp_path)
    assert summary["demonstrated"] is True and summary["formal_score"] is None
    assert operations.count("model.close") == operations.count("backend.close") == 2
