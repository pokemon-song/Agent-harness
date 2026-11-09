"""The offline example must prove persistence without network or model scoring."""

import importlib.util
import json
from pathlib import Path
import socket

import pytest


def demo_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_memory_demo.py"
    spec = importlib.util.spec_from_file_location("offline_memory_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_demo_public_motion_reopen_and_readonly_without_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Offline memory demonstration must never contact a service")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    from go2_demo.models import APIModel
    monkeypatch.setattr(APIModel, "from_env", no_network)
    monkeypatch.setattr(APIModel, "decide", no_network)
    demo = demo_module()
    directory = tmp_path / "fresh-offline-demo"
    result = demo.run_demo(directory)
    assert result["success"] is True, result.get("error")
    assert result["is_llm_result"] is False and result["api_requests"] == 0
    assert result["model_score"] is None and result["simulator_used"] is False
    assert result["scripted_test_double"] is True and result["scripted_decisions"] == 10
    assert all(result["checks"].values())
    first, second = result["episodes"]
    assert [row["calls"][0]["name"] for row in first["events"]] == [
        "move_relative", "memory_save", "memory_working_note", "observe", "abort"]
    assert [row["calls"][0]["name"] for row in second["events"]] == [
        "memory_search", "memory_read", "memory_working_note", "observe", "abort"]
    assert first["events"][0]["public_evidence_ref"]
    assert first["events"][1]["public_evidence_ref"] is None
    assert second["initial_working_note"] == ""
    assert second["final_working_note"] == demo.SECOND_NOTE
    assert result["readonly_database_sha256_before"] == result["readonly_database_sha256_after"]
    assert result["memory_snapshot_sha256_before"] == result["memory_snapshot_sha256_after"]
    assert json.loads((directory / "validation.json").read_text()) == result
    assert not list(directory.rglob("metrics.json"))


def test_offline_demo_does_not_overwrite_existing_evidence(tmp_path):
    demo = demo_module()
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "validation.json"
    marker.write_bytes(b"existing evidence\n")
    with pytest.raises(FileExistsError):
        demo.run_demo(directory)
    assert marker.read_bytes() == b"existing evidence\n"
