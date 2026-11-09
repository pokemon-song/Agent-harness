"""One visible model loop for memory and motion; no real API requests."""

import json

from go2_demo import __version__
from go2_demo.runner import run_episode
from go2_demo.types import Decision, ToolCall
from test_runner import EpisodeBackend, EpisodeModel


class RememberModel:
    model = "local-memory-test"

    def __init__(self):
        self.turn = 0
        self.closed = False

    def decide(self, observation, image, tools, history, **kwargs):
        context = observation["agent_context"]
        assert context["budget"]["remaining_calls"] == 8 - self.turn
        assert "PRIVATE-REASONING" not in json.dumps(context)
        if self.turn == 0:
            name, args = "observe", {}
        elif self.turn == 1:
            name, args = "memory_save", {"text": "fixture lesson: use measured motion feedback",
                                         "evidence_refs": [context["public_evidence_refs"][-1]]}
        elif self.turn == 2:
            name, args = "memory_working_note", {"text": "fixture working note retained across history truncation"}
        elif self.turn == 3:
            name, args = "memory_search", {"query": "fixture lesson"}
        else:
            assert "fixture working note" in json.dumps(context)
            assert len(history) == 1  # pinned note survives normal rolling history
            name, args = "finish", {}
        self.turn += 1
        return Decision([ToolCall(f"memory-step-{self.turn}", name, args)],
                        model=self.model, reasoning_content="PRIVATE-REASONING-NEVER-PERSIST",
                        usage={"total_tokens": 10})

    def close(self):
        self.closed = True


def test_memory_and_robot_tools_share_budget_and_archive_public_evidence(tmp_path):
    model = RememberModel()
    result = run_episode(EpisodeBackend(), model, mode="api", run_dir=tmp_path / "run",
                         max_model_calls=8, history_turns=1, memory_mode="read_write",
                         memory_dir=tmp_path / "memory")
    assert result["success"] and result["external_memory"]
    assert result["model_calls"] == 5 and result["memory_tool_calls"] == 3
    assert result["usage"]["total_tokens"] == 50
    assert result["experiment_kind"] == "memory-demo-v1"
    initial = json.loads((tmp_path / "run/memory-initial.json").read_text())
    final = json.loads((tmp_path / "run/memory-final.json").read_text())
    assert initial["sha256"] != final["sha256"]
    assert "fixture lesson" in json.dumps(final)
    for excluded in ("PRIVATE-REASONING", "hidden_target", "target_xy_m", "fixture working note"):
        assert excluded not in json.dumps(final)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["task_agent"] and manifest["external_memory"]
    assert manifest["application_version"] == __version__
    assert manifest["memory_initial_sha256"] == initial["sha256"]
    assert model.closed


def test_read_only_reopens_saved_memory_without_changing_persistent_snapshot(tmp_path):
    run_episode(EpisodeBackend(), RememberModel(), mode="api", run_dir=tmp_path / "write",
                max_model_calls=8, history_turns=1, memory_mode="read_write", memory_dir=tmp_path / "memory")
    before = json.loads((tmp_path / "write/memory-final.json").read_text())["sha256"]

    class RecallModel(EpisodeModel):
        def __init__(self):
            super().__init__()
            self.turn = 0

        def decide(self, observation, image, tools, history, **kwargs):
            names = {t["function"]["name"] for t in tools}
            assert "memory_save" not in names and "memory_revise" not in names
            assert "fixture working note" not in json.dumps(observation)
            if self.turn == 0:
                assert "fixture lesson" not in json.dumps(observation)  # no forced retrieval
                self.turn += 1
                return Decision([ToolCall("recall", "memory_search", {"query": "fixture lesson"})])
            assert "fixture lesson" in json.dumps(history[-1]["results"])
            return Decision([ToolCall("done", "finish", {})])

    result = run_episode(EpisodeBackend(), RecallModel(), mode="api", run_dir=tmp_path / "read",
                         memory_mode="read_only", memory_dir=tmp_path / "memory")
    assert result["success"]
    assert result["memory_initial_sha256"] == result["memory_final_sha256"] == before
    assert result["model_calls"] == 2


def test_memory_off_does_not_create_a_database_or_memory_context(tmp_path):
    class NoMemoryModel(EpisodeModel):
        def decide(self, observation, image, tools, history, **kwargs):
            assert "agent_context" not in observation
            assert not any(t["function"]["name"].startswith("memory_") for t in tools)
            return super().decide(observation, image, tools, history, **kwargs)

    result = run_episode(EpisodeBackend(), NoMemoryModel(), mode="api", run_dir=tmp_path / "run",
                         memory_mode="off", memory_dir=tmp_path / "unused")
    assert result["success"] and not (tmp_path / "unused").exists()


def test_missing_read_only_memory_fails_without_calling_model(tmp_path):
    result = run_episode(EpisodeBackend(), EpisodeModel(after_decision=lambda: 1 / 0),
                         mode="api", run_dir=tmp_path / "run", memory_mode="read_only",
                         memory_dir=tmp_path / "absent")
    assert not result["success"] and result["model_calls"] == 0
    assert result["external_memory"] and result["memory_mode"] == "read_only"
    assert not (tmp_path / "absent").exists()


def test_memory_request_still_consumes_last_model_call(tmp_path):
    class SearchModel(EpisodeModel):
        def decide(self, *args, **kwargs):
            return Decision([ToolCall("search", "memory_search", {"query": "nothing"})])

    result = run_episode(EpisodeBackend(), SearchModel(), mode="api", run_dir=tmp_path / "run",
                         max_model_calls=1, memory_mode="read_write", memory_dir=tmp_path / "memory")
    assert result["termination"] == "budget_exhausted" and result["model_calls"] == 1
    assert result["memory_tool_calls"] == 1 and not result["success"]
