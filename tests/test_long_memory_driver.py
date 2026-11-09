"""Control conditions must differ only in accessible memory, never the answer prompt."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go2_demo.harness import Harness
from go2_demo.memory import MemoryStore
from go2_demo.memory_mission import DESTINATION_XY, delivery_scenario, recall_instruction
from go2_demo.task_evaluation import TaskEvaluationState, TaskEvaluator


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_long_memory_mission.py"
    spec = importlib.util.spec_from_file_location("long_memory_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_copy_has_identical_memory_but_empty_control_has_no_order(driver, tmp_path):
    source = tmp_path / "briefing/memory.sqlite3"
    store = MemoryStore(source, scope=driver.scope(), mode="read_write", episode_id="briefing")
    assert store.register_public_evidence("order-evidence", {
        "source": "public_harness_turn", "observation": {"task": "Order fixture: C, 012345"}})["status"] == "succeeded"
    assert store.save("Order fixture: C, 012345", ["order-evidence"], call_id="save")["status"] == "succeeded"
    store.close()
    source_sha = driver.file_hash(source)
    full, empty = tmp_path / "full/memory.sqlite3", tmp_path / "empty/memory.sqlite3"
    driver.copy_memory(source, full)
    driver.initialize_empty(empty)
    assert driver.memory_snapshot(source) == driver.memory_snapshot(full)
    assert driver.memory_snapshot(empty)["content"]["records"] == []
    assert "012345" not in json.dumps(driver.memory_snapshot(empty))
    assert driver.file_hash(source) == source_sha
    with pytest.raises(FileExistsError):
        driver.copy_memory(source, full)


def test_no_automatic_answer_is_inserted_when_model_fails_to_save(driver, monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(driver, "check_frozen", lambda config: None)

    def fake_worker(directory, phase, key):
        called.append(phase)
        driver.initialize_empty(directory / "memory" / phase / "memory.sqlite3")
        driver.write_json(directory / phase / "metrics.json", {"success": True, "termination": "finish", "model_calls": 2})
        driver.write_json(directory / phase / "worker.json", {"pid": 1, "cleanup_errors": []})
        return 0

    monkeypatch.setattr(driver, "launch_worker", fake_worker)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 1
    summary = driver.read_json(next(tmp_path.glob("*/summary.json")))
    assert called == ["briefing"]
    assert summary["not_run"] == ["with_memory", "empty_memory"]
    assert summary["mission_success"] is False
    assert summary["demonstrated_memory_dependency"] is False


def test_worker_infrastructure_failure_is_preserved_without_retry(driver, monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(driver, "check_frozen", lambda config: None)

    def fake_worker(directory, phase, key):
        called.append(phase)
        driver.write_json(directory / phase / "metrics.json", {
            "success": False, "termination": "model_error", "error": {"code": "MODEL_AUTH_ERROR"}, "model_calls": 1})
        driver.write_json(directory / phase / "worker.json", {"pid": 1, "cleanup_errors": []})
        return 1

    monkeypatch.setattr(driver, "launch_worker", fake_worker)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 1
    summary = driver.read_json(next(tmp_path.glob("*/summary.json")))
    assert called == ["briefing"] and summary["api_requests"] == 1
    assert summary["phases"][0]["metrics"]["error"]["code"] == "MODEL_AUTH_ERROR"
    assert summary["stopped_reason"] == "Infrastructure failure; no automatic retries"


def worker_config(driver, directory):
    order = {"order_id": "fixture-order", "destination": "C", "receipt_code": "012345"}
    driver.write_json(directory / "order.private.json", order)
    config = {"order_id": order["order_id"], "order_file_sha256": driver.file_hash(directory / "order.private.json"),
              "budgets": driver.BUDGETS}
    driver.write_json(directory / "experiment.json", config)
    return config


def test_cleanup_exception_does_not_overwrite_archived_metrics(driver, monkeypatch, tmp_path):
    from go2_demo.models import APIModel
    from go2_demo import runner
    from go2_demo.robots import registry
    worker_config(driver, tmp_path)
    monkeypatch.setattr(driver, "check_frozen", lambda config: None)
    closed = []
    model = SimpleNamespace(close=lambda: closed.append("model"))
    backend = SimpleNamespace(stop=lambda: closed.append("stop"), close=lambda: closed.append("backend"))
    monkeypatch.setattr(APIModel, "from_env", lambda **kwargs: model)
    monkeypatch.setattr(registry, "create_backend", lambda *args, **kwargs: backend)
    original = {"success": True, "termination": "finish", "model_calls": 17, "usage": {"total_tokens": 12345}}

    def fail_after_archive(*args, run_dir, **kwargs):
        driver.write_json(run_dir / "metrics.json", original)
        raise RuntimeError("simulated close failure")

    monkeypatch.setattr(runner, "run_episode", fail_after_archive)
    assert driver.worker(tmp_path, "briefing") == 1
    assert driver.read_json(tmp_path / "briefing/metrics.json") == original
    result = driver.phase_result(tmp_path, "briefing", 1)
    assert result["worker"]["error"]["message"] == "simulated close failure"
    assert driver.infrastructure_failed(result)
    assert closed == ["stop", "backend", "model"]


def test_order_tampering_fails_before_credentials_are_loaded(driver, monkeypatch, tmp_path):
    from go2_demo.models import APIModel
    worker_config(driver, tmp_path)
    driver.write_json(tmp_path / "order.private.json", {"order_id": "fixture-order", "destination": "A", "receipt_code": "987654"})
    monkeypatch.setattr(driver, "check_frozen", lambda config: None)
    monkeypatch.setattr(APIModel, "from_env", lambda **kwargs: pytest.fail("Tampered order reached credential loader"))
    assert driver.worker(tmp_path, "with_memory") == 1
    m = driver.read_json(tmp_path / "with_memory/metrics.json")
    assert not m["success"] and "Private order changed" in m["error"]["message"]


def test_bad_event_archive_preserves_known_metrics_and_stops_experiment(driver, tmp_path):
    driver.write_json(tmp_path / "briefing/metrics.json", {"success": True, "termination": "finish", "model_calls": 17})
    driver.write_json(tmp_path / "briefing/worker.json", {"pid": 1, "cleanup_errors": []})
    (tmp_path / "briefing/events.jsonl").write_text('{"event":')
    result = driver.phase_result(tmp_path, "briefing", 0)
    assert result["metrics"]["model_calls"] == 17 and result["artifact_error"]
    assert driver.infrastructure_failed(result)


def test_fixed_b_geometry_placeholder_does_not_override_hidden_delivery_target():
    scene = delivery_scenario()
    state = TaskEvaluationState(np.array([2.2, .8, .34]), 0., np.zeros(2), 0., False, 0, [], 1.)
    a = TaskEvaluator(scene, DESTINATION_XY["A"], lambda: state)
    b = TaskEvaluator(scene, DESTINATION_XY["B"], lambda: state)
    assert a.public_task_spec == b.public_task_spec
    assert a.evaluate()["success"] is True
    assert b.evaluate()["success"] is False


def test_hidden_destination_changes_neither_real_rgb_nor_public_observation(monkeypatch):
    from go2_demo.models import APIModel
    from go2_demo.robots.registry import create_backend

    monkeypatch.setattr(APIModel, "from_env", lambda **kwargs: pytest.fail("No credentials or API in this physics test"))
    root = Path(__file__).resolve().parents[1]
    if not (root / "assets/go2_rl_lab/deploy/pre_train/go2/go2_cts_150k.pt").exists():
        pytest.skip("Checksum-verified local Go2 assets are unavailable")
    backend = create_backend("go2", "mujoco", asset_dir=root / "assets", seed=0, scenario=delivery_scenario())
    try:
        observations, images = [], []
        for letter in "ABC":
            backend.evaluator = TaskEvaluator(backend.scenario, DESTINATION_XY[letter], backend._evaluation_state)
            harness = Harness(backend, recall_instruction("fixture-order"))
            observation = harness.observation()
            observation.pop("image_id")  # Capture serial increments without changing the physical state.
            observations.append(observation)
            images.append(harness.frame.rgb.copy())
        assert observations[0] == observations[1] == observations[2]
        assert np.array_equal(images[0], images[1]) and np.array_equal(images[1], images[2])
    finally:
        backend.close()
