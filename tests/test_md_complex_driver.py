"""Memory lineage, read-only controls and accounting for the MD complex driver."""
import importlib.util
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
from PIL import Image

from go2_demo.md_memory.store import MDMemoryStore, StoreError


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[1] / "scripts/run_md_complex_memory_mission.py"
    spec = importlib.util.spec_from_file_location("md_complex_driver_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    from go2_demo.models import APIModel

    def forbidden(*args, **kwargs):
        raise AssertionError("Driver tests cannot request network or API credentials")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(APIModel, "from_env", forbidden)


SCOPE = "world-fixture-order"


def remember(directory, phase, text):
    store = MDMemoryStore(directory / "memory" / phase, scope=SCOPE)
    before = store.snapshot()
    relative = f"evidence/event-{before['latest_event_seq']+1}.png"
    image = store.root / relative
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(before["revision"] * 20, 0, 0)).save(image)
    event = store.append_event("user_instruction", {"notice": text, "phase": phase}, image_path=relative, critical=True)
    store.commit(before["revision"], event["seq"], [{"path": "tasks/current_order.md", "body": text,
        "evidence_ids": [event["id"]], "status": "confirmed"}], call_id=f"{phase}-{event['seq']}")
    return store.snapshot()


def test_full_stale_and_empty_use_exact_real_md_parent_directories(driver, tmp_path):
    assert driver.prepare_memory(tmp_path, "dispatch", SCOPE)
    first = remember(tmp_path, "dispatch", "Initial R1 left")
    first_hashes = driver.tree_hashes(tmp_path / "memory/dispatch")
    assert driver.prepare_memory(tmp_path, "fragments", SCOPE)
    assert driver.tree_hashes(tmp_path / "memory/fragments") == first_hashes
    second = remember(tmp_path, "fragments", "R1 left plus second fragment and mapping")
    second_hashes = driver.tree_hashes(tmp_path / "memory/fragments")
    assert driver.prepare_memory(tmp_path, "revision", SCOPE)
    assert driver.tree_hashes(tmp_path / "memory/revision") == second_hashes
    third = remember(tmp_path, "revision", "ACTIVE R2 right; ARCHIVED R1 left")
    third_hashes = driver.tree_hashes(tmp_path / "memory/revision")
    assert driver.prepare_memory(tmp_path, "full", SCOPE)
    assert driver.prepare_memory(tmp_path, "stale", SCOPE)
    assert driver.tree_hashes(tmp_path / "memory/full") == third_hashes
    assert driver.tree_hashes(tmp_path / "memory/stale") == second_hashes
    assert driver.prepare_memory(tmp_path, "empty", SCOPE)
    empty = MDMemoryStore(tmp_path / "memory/empty", scope=SCOPE).snapshot()
    assert empty["files"] == [] and empty["latest_event_seq"] == 0
    for phase, snapshot, expected in (("dispatch", first, first_hashes), ("fragments", second, second_hashes),
                                      ("revision", third, third_hashes), ("full", third, third_hashes),
                                      ("stale", second, second_hashes)):
        store = MDMemoryStore(tmp_path / "memory" / phase, scope=SCOPE)
        assert store.snapshot() == snapshot
        store.read("tasks/current_order.md")
        store.search("R1")
        assert driver.tree_hashes(store.root) == expected
    with pytest.raises(FileExistsError):
        driver.prepare_memory(tmp_path, "full", SCOPE)


def test_missing_or_unpublished_memory_does_not_fabricate_next_phase(driver, tmp_path):
    assert not driver.prepare_memory(tmp_path, "fragments", SCOPE)
    assert driver.prepare_memory(tmp_path, "dispatch", SCOPE)
    store = MDMemoryStore(tmp_path / "memory/dispatch", scope=SCOPE)
    store.append_event("user_instruction", {"notice": "Not yet processed"}, critical=True)
    assert not driver.prepare_memory(tmp_path, "fragments", SCOPE)
    assert not (tmp_path / "memory/fragments").exists()


def test_mismatched_world_scope_cannot_copy_parent(driver, tmp_path):
    remember(tmp_path, "dispatch", "A published fact")
    with pytest.raises(StoreError) as error:
        driver.prepare_memory(tmp_path, "fragments", "world-another-order")
    assert error.value.code == "SCOPE_MISMATCH"
    assert not (tmp_path / "memory/fragments").exists()


@pytest.mark.parametrize("condition", ["full", "stale"])
def test_readonly_final_refuses_parent_with_pending_critical_input(driver, tmp_path, condition):
    parent = driver.PARENTS[condition]
    remember(tmp_path, parent, "Previously published instruction")
    store = MDMemoryStore(tmp_path / "memory" / parent, scope=SCOPE)
    store.append_event("user_instruction", {"notice": "New instruction not yet processed"}, critical=True)
    before = driver.tree_hashes(store.root)
    assert not driver.prepare_memory(tmp_path, condition, SCOPE)
    assert not (tmp_path / "memory" / condition).exists()
    assert driver.tree_hashes(store.root) == before


def write_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")


def test_accounting_combines_roles_without_counting_tool_results_twice(driver, tmp_path):
    write_events(tmp_path / "dispatch/events.jsonl", [
        {"event": "model_input"},
        {"event": "model_response", "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        {"event": "tool_results", "usage": {"total_tokens": 999}},
        {"event": "model_input"},  # Timed out: attempted but no response.
    ])
    write_events(tmp_path / "dispatch/memory-agent.jsonl", [
        {"event": "model_request"},
        {"event": "model_response", "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23}},
        {"event": "tool_results"},
        {"event": "model_request"},
        {"event": "model_response", "usage": {"prompt_tokens": 4}},  # Partial usage stays partial.
    ])
    result = driver.model_accounting(tmp_path)
    assert result["api_requests"] == 4
    assert result["complete_responses"] == 3
    assert result["known_usage"] == {"prompt_tokens": 34, "completion_tokens": 5, "total_tokens": 35}
    assert result["unknown_usage_attempts"] == 2
    assert result["usage_complete"] is False


def test_accounting_rejects_boolean_token_counts_as_complete(driver, tmp_path):
    write_events(tmp_path / "dispatch/events.jsonl", [
        {"event": "model_input"},
        {"event": "model_response", "usage": {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3}},
    ])
    result = driver.model_accounting(tmp_path)
    assert result["known_usage"]["prompt_tokens"] == 0
    assert result["unknown_usage_attempts"] == 1 and not result["usage_complete"]


def test_accounting_retains_known_records_after_truncated_json_line(driver, tmp_path):
    path = tmp_path / "dispatch/events.jsonl"
    write_events(path, [{"event": "model_input"}, {"event": "model_response",
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}])
    with path.open("a") as stream:
        stream.write('{"event":"model_inp')
    result = driver.model_accounting(tmp_path)
    assert result["api_requests"] == 1 and result["complete_responses"] == 1
    assert result["known_usage"]["total_tokens"] == 12
    assert not result["usage_complete"]
    assert result["accounting_errors"]


def test_source_and_contract_changes_rejected_before_launch(driver, tmp_path, monkeypatch):
    contract = tmp_path / "contract.private.json"
    driver.write_json(contract, {"order_id": "fixture"})
    monkeypatch.setattr(driver, "frozen_sources", lambda: {"fixture.py": "source-digest"})
    monkeypatch.setattr(driver, "git_identity", lambda root: {"commit": "fixed", "dirty": False})
    config = {"source_sha256": driver.frozen_sources(), "git": driver.git_identity(None),
              "contract_sha256": driver.sha(contract)}
    driver.check_frozen(tmp_path, config)
    driver.write_json(contract, {"order_id": "changed"})
    with pytest.raises(RuntimeError, match="Contract changed"):
        driver.check_frozen(tmp_path, config)
    config["contract_sha256"] = driver.sha(contract)
    monkeypatch.setattr(driver, "frozen_sources", lambda: {"fixture.py": "changed"})
    with pytest.raises(RuntimeError, match="Source or Git changed"):
        driver.check_frozen(tmp_path, config)


def test_default_budgets_are_frozen_and_startup_failure_preserves_not_run(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "git_identity", lambda root: {"commit": "fixed", "dirty": False})
    monkeypatch.setattr(driver, "frozen_sources", lambda: {"fixture.py": "source-digest"})
    launched = []

    def failed_worker(command, **kwargs):
        phase = command[command.index("--worker")+1]
        directory = Path(command[command.index("--experiment-dir")+1])
        launched.append(phase)
        driver.write_json(directory / phase / "metrics.json", {"success": False, "termination": "startup_error"})
        driver.write_json(directory / phase / "worker.json", {"pid": 1, "cleanup_errors": [],
            "error": {"code": "FIXTURE_STARTUP_FAILURE"}, "readonly_memory_unchanged": True})
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(driver.subprocess, "run", failed_worker)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 1
    directory = next(tmp_path.glob("*-md-complex-memory-*"))
    config = driver.read_json(directory / "experiment.json")
    summary = driver.read_json(directory / "summary.json")
    assert launched == ["dispatch"]
    assert config["history_turns"] == 2
    assert config["max_output_tokens"] == 16384 and config["api_timeout_s"] == 180
    assert config["memory_calls_per_learning_phase"] == 12
    calls = sum(config["budgets"][phase if phase in driver.LEARNING else "recall"]["max_model_calls"] for phase in driver.PHASES)
    assert calls == 280 and calls + 3 * config["memory_calls_per_learning_phase"] == 316
    assert summary["not_run"] == list(driver.PHASES[1:])
    assert not summary["mission_success"]
    assert not summary["demonstrated_memory_dependency"] and not summary["demonstrated_update_dependency"]


def test_full_readonly_violation_cannot_leave_primary_mission_successful(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "git_identity", lambda root: {"commit": "fixed", "dirty": False})
    monkeypatch.setattr(driver, "frozen_sources", lambda: {"fixture.py": "source-digest"})
    launched = []

    def fake_worker(command, **kwargs):
        phase = command[command.index("--worker")+1]
        directory = Path(command[command.index("--experiment-dir")+1])
        config = driver.read_json(directory / "experiment.json")
        launched.append(phase)
        memory = directory / "memory" / phase
        before = driver.tree_hashes(memory)
        if phase in driver.LEARNING:
            store = MDMemoryStore(memory, scope=config["memory_scope"])
            snapshot = store.snapshot()
            event = store.append_event("user_instruction", {"phase": phase}, critical=True)
            store.commit(snapshot["revision"], event["seq"], [{"path": "tasks/order.md", "body": f"Published {phase}",
                "evidence_ids": [event["id"]], "status": "confirmed"}], call_id=phase)
        else:
            (memory / "unexpected-write.txt").write_text("read-only violation")
        after = driver.tree_hashes(memory)
        driver.write_json(directory / phase / "metrics.json", {"success": True, "termination": "finish",
                                                               "memory_agent_stopped": True})
        driver.write_json(directory / phase / "worker.json", {"pid": len(launched), "cleanup_errors": [], "error": None,
            "readonly_memory_unchanged": phase in driver.LEARNING or before == after,
            "memory_before_digest": driver.digest(before), "memory_after_digest": driver.digest(after)})
        return SimpleNamespace(returncode=0 if phase in driver.LEARNING else 1)

    monkeypatch.setattr(driver.subprocess, "run", fake_worker)
    assert driver.main(["--runs-dir", str(tmp_path)]) == 1
    directory = next(tmp_path.glob("*-md-complex-memory-*"))
    summary = driver.read_json(directory / "summary.json")
    assert launched == [*driver.LEARNING, "full"]
    assert not summary["mission_success"]
    assert summary["not_run"] == ["stale", "empty"]
    assert not summary["demonstrated_memory_dependency"] and not summary["demonstrated_update_dependency"]


def audit_fixture(driver, directory):
    results = []
    driver.write_json(directory / "experiment.json", {"memory_scope": SCOPE})
    manifest = {key: "same" for key in ("task", "system_prompt_sha256", "tools_sha256", "budgets",
                                       "api_settings", "robot_profile_sha256", "code_sha256")}
    for index, phase in enumerate(driver.PHASES):
        driver.prepare_memory(directory, phase, SCOPE)
        before = driver.tree_hashes(directory / "memory" / phase)
        store = MDMemoryStore(directory / "memory" / phase, scope=SCOPE)
        driver.write_json(directory / phase / "memory-initial.json", {"storage": "versioned_markdown", **store.snapshot()})
        if phase in driver.LEARNING:
            remember(directory, phase, f"Record from {phase}")
        after = driver.tree_hashes(directory / "memory" / phase)
        driver.write_json(directory / phase / "memory-final.json", {"storage": "versioned_markdown", **store.snapshot()})
        worker = {"pid": index+10, "readonly_memory_unchanged": True, "cleanup_errors": [], "error": None,
                  "memory_before_digest": driver.digest(before), "memory_after_digest": driver.digest(after)}
        driver.write_json(directory / phase / "worker.json", worker)
        driver.write_json(directory / phase / "manifest.json", {**manifest,
            "memory_mode": "read_write" if phase in driver.LEARNING else "read_only"})
        image = directory / phase / "frames/first.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (0, 0, 0)).save(image)
        write_events(directory / phase / "events.jsonl", [{"event": "model_input", "history_turns": 0,
            "image_path": "frames/first.png", "observation": {"task": "same", "robot_state": {"idle": True},
            "memory": {"selected_revision": index, "published_revision": index, "scope": SCOPE,
                       "mode": "read_only", "files": [], "read_receipts": [], "required_reads": [],
                       "needs_refresh": False, "pending_critical_events": []},
            "agent_context": {"episode_id": phase, "scene_id": "same-final-scene", "remaining_wall_s": 1200-index,
                              "remaining_calls": 60, "coordinate_reference": "this_episode_start"}}}])
        results.append({"phase": phase, "worker": worker, "metrics": {"success": True}})
    return results


def test_final_audit_detects_changed_initial_frame_and_readonly_violation(driver, tmp_path):
    results = audit_fixture(driver, tmp_path)
    assert driver.final_audit(tmp_path, results)["passed"]
    Image.new("RGB", (4, 4), (255, 0, 0)).save(tmp_path / "empty/frames/first.png")
    assert not driver.final_audit(tmp_path, results)["passed"]
    Image.new("RGB", (4, 4), (0, 0, 0)).save(tmp_path / "empty/frames/first.png")
    results[-1]["worker"]["readonly_memory_unchanged"] = False
    assert not driver.final_audit(tmp_path, results)["passed"]


def test_final_audit_missing_control_is_not_a_pass(driver, tmp_path):
    assert driver.final_audit(tmp_path, [{"phase": "full"}, {"phase": "stale"}]) is None


def test_final_audit_all_missing_manifest_field_is_not_equal_configuration(driver, tmp_path):
    results = audit_fixture(driver, tmp_path)
    for phase in driver.CONDITIONS:
        path = tmp_path / phase / "manifest.json"
        manifest = driver.read_json(path)
        del manifest["tools_sha256"]
        driver.write_json(path, manifest)
    assert not driver.final_audit(tmp_path, results)["passed"]


def test_final_audit_detects_changed_public_initial_observation(driver, tmp_path):
    results = audit_fixture(driver, tmp_path)
    path = tmp_path / "empty/events.jsonl"
    event = json.loads(path.read_text())
    event["observation"]["robot_state"]["idle"] = False
    write_events(path, [event])
    assert not driver.final_audit(tmp_path, results)["passed"]


def test_final_audit_detects_parent_changed_after_copy(driver, tmp_path):
    results = audit_fixture(driver, tmp_path)
    (tmp_path / "memory/fragments/unexpected-change.txt").write_text("Parent changed after stale copy")
    assert not driver.final_audit(tmp_path, results)["passed"]


def test_worker_preserves_archived_metrics_when_episode_cleanup_raises(driver, tmp_path, monkeypatch):
    from go2_demo.models import APIModel
    from go2_demo.md_memory import navigation
    from go2_demo.robots import registry

    contract = driver.make_contract()
    config = {"memory_scope": SCOPE, "history_turns": 2, "api_timeout_s": 180,
              "max_output_tokens": 16384, "recording": False,
              "memory_calls_per_learning_phase": 12, "budgets": driver.BUDGETS}
    driver.write_json(tmp_path / "contract.private.json", contract)
    driver.write_json(tmp_path / "experiment.json", config)
    driver.prepare_memory(tmp_path, "dispatch", SCOPE)
    monkeypatch.setattr(driver, "check_frozen", lambda *args: None)
    closed, created = [], []

    def model_factory(**kwargs):
        assert kwargs["timeout_s"] == 180 and kwargs["max_output_tokens"] == 16384
        assert kwargs["history_turns"] == 2
        label = "planner" if not created else "memory"
        model = SimpleNamespace(close=lambda: closed.append(label))
        created.append(model)
        return model

    monkeypatch.setattr(APIModel, "from_env", model_factory)
    backend = SimpleNamespace(stop=lambda: closed.append("stop"), close=lambda: closed.append("backend"))
    monkeypatch.setattr(registry, "create_backend", lambda *args, **kwargs: backend)
    archived = {"success": True, "termination": "finish", "model_calls": 7,
                "api_requests_total": 9, "path_length_m": 3.2, "usage_total": {"total_tokens": 1234}}

    def fail_after_archive(*args, run_dir, **kwargs):
        driver.write_json(run_dir / "metrics.json", archived)
        raise OSError("PRIVATE_LOG_CLOSE_DETAIL")

    monkeypatch.setattr(navigation, "run_md_episode", fail_after_archive)
    assert driver.worker(tmp_path, "dispatch", None) == 1
    assert driver.read_json(tmp_path / "dispatch/metrics.json") == archived
    worker = driver.read_json(tmp_path / "dispatch/worker.json")
    assert worker["error"]["type"] == "OSError"
    assert "PRIVATE_LOG_CLOSE_DETAIL" not in json.dumps(worker)
    assert closed == ["stop", "backend", "planner", "memory"]
