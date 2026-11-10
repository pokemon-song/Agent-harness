"""Real snapshot lineage and final-score attribution for the harder mission."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from go2_demo.complex_memory_mission import make_contract, recall_instruction
from go2_demo.complex_world import DESTINATION_XY, final_scenario
from go2_demo.memory import MemoryScope, MemoryStore
from go2_demo.robots.registry import get_profile


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_complex_memory_mission.py"
    spec = importlib.util.spec_from_file_location("complex_memory_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_fact(directory, phase, text):
    scope = MemoryScope.from_profile(get_profile("go2", "mujoco"))
    store = MemoryStore(directory / "memory" / phase / "memory.sqlite3", scope=scope,
                        mode="read_write", episode_id=phase)
    try:
        ref = phase + "-public"
        assert store.register_public_evidence(ref, {"source": "public_harness_turn",
                                                    "observation": {"task": text}})["status"] == "succeeded"
        assert store.save(text, [ref], call_id=phase + "-save")["status"] == "succeeded"
        return store.snapshot()
    finally:
        store.close()


def test_stale_is_exact_pre_revision_snapshot_and_full_keeps_all_agent_records(driver, tmp_path):
    initial = save_fact(tmp_path, "dispatch", "fixture initial dispatch")
    assert driver.prepare_memory(tmp_path, "fragments")
    second = save_fact(tmp_path, "fragments", "fixture second code fragment")
    assert driver.prepare_memory(tmp_path, "revision")
    updated = save_fact(tmp_path, "revision", "fixture revised destination")
    for phase, expected in (("full", updated), ("stale", second)):
        assert driver.prepare_memory(tmp_path, phase)
        assert driver.memory_snapshot(tmp_path / "memory" / phase / "memory.sqlite3") == expected
    assert driver.prepare_memory(tmp_path, "empty")
    assert driver.memory_snapshot(tmp_path / "memory/empty/memory.sqlite3")["content"]["records"] == []
    for phase, expected in (("dispatch", initial), ("fragments", second), ("revision", updated)):
        assert driver.memory_snapshot(tmp_path / "memory" / phase / "memory.sqlite3") == expected
    with pytest.raises(FileExistsError):
        driver.prepare_memory(tmp_path, "full")


def test_missing_saved_memory_stops_without_manufacturing_an_answer(driver, tmp_path, monkeypatch):
    phases = []
    monkeypatch.setattr(driver, "check_frozen", lambda config: None)

    def fake_launch(directory, phase, key):
        phases.append(phase)
        driver.initialize_empty(directory / "memory" / phase / "memory.sqlite3")
        driver.write_json(directory / phase / "metrics.json", {"success": True, "termination": "finish", "model_calls": 3})
        driver.write_json(directory / phase / "worker.json", {"pid": 1, "cleanup_errors": [], "error": None})
        return 0

    monkeypatch.setattr(driver, "launch_worker", fake_launch)
    assert driver.main(["--runs-dir", str(tmp_path), "--no-recording"]) == 1
    summary = driver.read_json(next(tmp_path.glob("*/summary.json")))
    assert phases == ["dispatch"]
    assert summary["api_requests"] == 3 and not summary["mission_success"]
    assert "No saved agent memory" in summary["stopped_reason"]
    assert summary["not_run"] == list(driver.PHASES[1:])


def worker_config(driver, directory):
    contract = make_contract()
    driver.write_json(directory / "contract.private.json", contract)
    driver.write_json(directory / "experiment.json", {"contract_sha256": driver.file_hash(directory / "contract.private.json"),
        "order_id": contract["order_id"], "budgets": driver.BUDGETS, "recording": False})
    return contract


def test_contract_tampering_fails_before_any_api_credentials(driver, tmp_path, monkeypatch):
    from go2_demo.models import APIModel
    c = worker_config(driver, tmp_path)
    c["new_route"] = c["old_route"]
    driver.write_json(tmp_path / "contract.private.json", c)
    monkeypatch.setattr(driver, "check_frozen", lambda config: None)
    monkeypatch.setattr(APIModel, "from_env", lambda **kwargs: pytest.fail("Tampering reached API setup"))
    assert driver.worker(tmp_path, "full") == 1
    assert "Private contract changed" in driver.read_json(tmp_path / "full/worker.json")["error"]["message"]


def test_worker_preserves_completed_metrics_when_cleanup_raises(driver, tmp_path, monkeypatch):
    from go2_demo.models import APIModel
    from go2_demo import runner
    from go2_demo.robots import registry
    worker_config(driver, tmp_path)
    monkeypatch.setattr(driver, "check_frozen", lambda config: None)
    closed = []
    monkeypatch.setattr(APIModel, "from_env", lambda **kwargs: SimpleNamespace(close=lambda: closed.append("model")))
    monkeypatch.setattr(registry, "create_backend", lambda *args, **kwargs:
        SimpleNamespace(stop=lambda: closed.append("stop"), close=lambda: closed.append("backend")))
    result = {"success": True, "termination": "finish", "model_calls": 12, "usage": {"total_tokens": 300}}

    def cleanup_error(*args, run_dir, **kwargs):
        driver.write_json(run_dir / "metrics.json", result)
        raise OSError("fixture archive close error")

    monkeypatch.setattr(runner, "run_episode", cleanup_error)
    assert driver.worker(tmp_path, "dispatch") == 1
    assert driver.read_json(tmp_path / "dispatch/metrics.json") == result
    archived = driver.phase_result(tmp_path, "dispatch", 1)
    assert driver.infrastructure_failed(archived)
    assert closed == ["stop", "backend", "model"]


def phase_result(phase, success=True, **metrics):
    return {"phase": phase, "metrics": {"success": success, "termination": "finish", "model_calls": 2, **metrics},
            "worker": {"cleanup_errors": [], "error": None}, "retrieved_nonempty_memory": phase == "full"}


def test_collision_or_missing_code_is_not_evidence_for_outdated_memory_effect(driver, tmp_path):
    c = make_contract()
    results = [phase_result(p) for p in driver.LEARNING + ("full",)]
    results += [phase_result("stale", False, termination="execution_failure"),
                phase_result("empty", False, termination="abort")]
    audit = {"passed": True, "full_stale_initial_memory_different": True}
    s = driver.summarize(tmp_path, results, c, audit, None, None)
    assert s["mission_success"] and s["demonstrated_memory_dependency"]
    assert not s["demonstrated_update_dependency"]
    submission = {"order_id": c["order_id"], "dispatch_revision": c["old_revision"],
        "destination": c["alias_to_destination"][c["old_alias"]], "route": c["old_route"],
        "receipt_code": c["receipt_code"]}
    results[-2] = phase_result("stale", False, evaluation={"task_extension": {"submission": submission}})
    s = driver.summarize(tmp_path, results, c, audit, None, None)
    assert s["demonstrated_update_dependency"] and s["stale_submitted_complete_old_order"]
    assert not driver.summarize(tmp_path, results, c, {"passed": True}, None, None)["demonstrated_update_dependency"]
    results[3]["retrieved_nonempty_memory"] = False
    assert not driver.summarize(tmp_path, results, c, audit, None, None)["demonstrated_update_dependency"]
    results[3]["retrieved_nonempty_memory"] = True
    submission["receipt_code"] = "bad"
    assert not driver.summarize(tmp_path, results, c, audit, None, None)["demonstrated_update_dependency"]
    results[0]["metrics"]["success"] = False
    assert not driver.summarize(tmp_path, results, c, {"passed": True}, None, None)["mission_success"]


def test_hidden_contract_target_does_not_change_real_world_or_public_criteria():
    from go2_demo.harness import Harness
    from go2_demo.robots.registry import create_backend
    from go2_demo.task_evaluation import TaskEvaluator
    root = Path(__file__).resolve().parents[1]
    if not (root / "assets/go2_rl_lab/deploy/pre_train/go2/go2_cts_150k.pt").exists():
        pytest.skip("Local Go2 assets unavailable")
    b = create_backend("go2", "mujoco", asset_dir=root / "assets", seed=0, scenario=final_scenario())
    try:
        images, observations = [], []
        for letter in "ABC":
            b.evaluator = TaskEvaluator(b.scenario, DESTINATION_XY[letter], b._evaluation_state)
            h = Harness(b, recall_instruction("fixture-order"))
            o = h.observation()
            o.pop("image_id")
            observations.append(o)
            images.append(h.frame.rgb.copy())
        assert observations[0] == observations[1] == observations[2]
        assert all(np.array_equal(images[0], rgb) for rgb in images[1:])
    finally:
        b.close()


@pytest.mark.parametrize("failed_control", ["stale", "empty"])
def test_control_cleanup_failure_cannot_count_as_memory_dependency(driver, tmp_path, failed_control):
    c = make_contract()
    submission = {"order_id": c["order_id"], "dispatch_revision": c["old_revision"],
        "destination": c["alias_to_destination"][c["old_alias"]], "route": c["old_route"],
        "receipt_code": c["receipt_code"]}
    results = [phase_result(p) for p in driver.LEARNING + ("full",)]
    results += [phase_result("stale", False, evaluation={"task_extension": {"submission": submission}}),
                phase_result("empty", False, termination="abort")]
    next(r for r in results if r["phase"] == failed_control)["worker"]["cleanup_errors"] = [{"error": "fixture"}]
    result = driver.summarize(tmp_path, results, c, {"passed": True, "full_stale_initial_memory_different": True}, None, None)
    assert result["mission_success"]
    assert not result["controls_infrastructure_valid"]
    assert not result["demonstrated_memory_dependency"]
    assert not result["demonstrated_update_dependency"]
