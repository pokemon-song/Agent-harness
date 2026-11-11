"""Dual-agent navigation invariants without a simulator or a paid model."""

from copy import deepcopy
import hashlib
import json
import threading
import time

import pytest

from go2_demo.md_memory.navigation import run_md_episode
from go2_demo.md_memory.store import MDMemoryStore
from go2_demo.models import ModelError
from go2_demo.types import Decision, ToolCall
from test_runner import EpisodeBackend


USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


def call(name, args=None, *, identifier=None, usage=None, **private):
    return Decision([ToolCall(identifier or name, name, args or {})],
                    usage=USAGE if usage is None else usage, **private)


class Model:
    model = "scripted-unit-fixture"

    def __init__(self, steps):
        self.steps = iter(steps)
        self.inputs = []
        self.closed = 0

    def decide(self, observation, image, tools, history, *, timeout_s):
        self.inputs.append((deepcopy(observation), deepcopy(history), deepcopy(tools), timeout_s))
        step = next(self.steps)
        if isinstance(step, BaseException):
            raise step
        return step(observation, history) if callable(step) else step

    def close(self):
        self.closed += 1


def change(event, body="The current dispatch is blue.", path="tasks/order.md"):
    return {"path": path, "body": body, "evidence_ids": [event["id"]], "status": "confirmed"}


def seed(store, body="The current dispatch is blue."):
    event = store.append_event("instruction", {"text": body})
    store.commit(0, event["seq"], [change(event, body)], call_id="seed")
    return event


def run(tmp_path, planner, *, store=None, backend=None, **kwargs):
    store = store or MDMemoryStore(tmp_path / "memory")
    backend = backend or EpisodeBackend()
    metrics = run_md_episode(backend, planner, run_dir=tmp_path / "run", store=store,
                             task="Perform the public patrol.", **kwargs)
    events = [json.loads(line) for line in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    return metrics, events, backend, store


def tool_results(events):
    return [result for event in events if event["event"] == "tool_results" for result in event["results"]]


def test_learning_notice_only_reaches_memory_model_and_index_is_not_full_document(tmp_path):
    notice = "PUBLIC_ORDER_CANARY R2 destination=yellow"

    def save(observation, history):
        event = observation["pending_events"][0]
        assert event["payload"]["text"] == notice
        assert event["payload"]["episode_id"] and event["payload"]["scene_id"]
        assert event["payload"]["observation"]["local_odometry"]["translation_m"] == [0., 0.]
        assert "hidden_target" not in json.dumps(event["payload"]["observation"])
        return call("commit_memory", {"changes": [change(event, notice)]})

    memory = Model([save, call("commit_memory", {"changes": []})])
    planner = Model([call("finish")])
    metrics, events, backend, store = run(tmp_path, planner, memory_model=memory, memory_notice=notice)
    assert metrics["success"]
    assert notice not in json.dumps(planner.inputs)
    assert planner.inputs[0][0]["memory"]["files"][0]["path"] == "tasks/order.md"
    assert store.read("tasks/order.md")["body"] == notice
    assert backend.closed is False and backend.stops >= 1
    assert planner.closed == memory.closed == 1
    assert metrics["api_requests_total"] == 3
    assert metrics["usage_total"]["total_tokens"] == 30
    assert metrics["usage_complete"]
    assert (tmp_path / "run" / "memory-initial.json").is_file()
    assert (tmp_path / "run" / "memory-agent.jsonl").is_file()
    assert (tmp_path / "run" / "memory-final.json").is_file()


def test_readonly_navigation_has_full_original_tools_and_does_not_write_store(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    seed(store)
    original = store.snapshot()
    file_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in store.root.rglob("*") if path.is_file()}
    planner = Model([call("move_relative", {"forward_m": .6, "lateral_m": 0}), call("finish")])
    metrics, events, backend, store = run(tmp_path, planner, store=store, read_only=True)
    assert metrics["success"] and backend.moves == [{"forward_m": .6, "lateral_m": 0}]
    names = {tool["function"]["name"] for tool in planner.inputs[0][2]}
    assert {"track_point", "get_geometry", "finish", "abort", "memory_read", "memory_refresh"} <= names
    assert "memory_save" not in names and "commit_memory" not in names
    assert store.snapshot() == original
    assert file_hashes == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in store.root.rglob("*") if path.is_file()}
    assert metrics["memory_agent"]["api_requests"] == 0
    assert not (tmp_path / "run" / "memory-agent.jsonl").exists()


def test_changed_read_document_requires_refresh_and_actual_reread(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    seed(store)

    def changed(observation, history):
        event = store.append_event("instruction", {"text": "Yellow now"})
        store.commit(1, event["seq"], [change(event, "Yellow now")], call_id="new")
        return call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="old-move")

    planner = Model([call("memory_read", {"path": "tasks/order.md"}), changed,
                     call("memory_refresh"),
                     call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="unread-move"),
                     call("memory_read", {"path": "tasks/order.md"}, identifier="new-read"),
                     call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="new-move"),
                     call("finish")])
    metrics, events, backend, _ = run(tmp_path, planner, store=store, read_only=True)
    assert metrics["success"] and len(backend.moves) == 1
    errors = [r.get("error_code") for r in tool_results(events)]
    assert "MEMORY_CHANGED" in errors and "MEMORY_REREAD_REQUIRED" in errors


def test_unrelated_document_revision_does_not_block_current_read_dependency(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    seed(store)

    def unrelated(observation, history):
        event = store.append_event("observation", {"text": "An unrelated box is visible"})
        store.commit(1, event["seq"], [change(event, "A box", "objects/box.md")], call_id="box")
        return call("move_relative", {"forward_m": .2, "lateral_m": 0})

    planner = Model([call("memory_read", {"path": "tasks/order.md"}), unrelated, call("finish")])
    metrics, _, backend, _ = run(tmp_path, planner, store=store, read_only=True)
    assert metrics["success"] and len(backend.moves) == 1


@pytest.mark.parametrize("path", ["tasks/order.md", "tasks/new-order.md"])
@pytest.mark.parametrize("act_before_refresh", [True, False])
def test_published_critical_notice_requires_read_of_previously_unread_document(tmp_path, path, act_before_refresh):
    store = MDMemoryStore(tmp_path / "memory")
    seed(store)
    store.append_event("feedback", {"text": "Older feedback remains pending"})

    def publish(observation, history):
        notice = store.append_event("instruction", {"text": "Use the revised order"}, critical=True)
        store.commit(1, notice["seq"], [change(notice, "Revised order", path)],
                     call_id="critical", event_seqs=[notice["seq"]])
        assert store.snapshot()["processed_event_seq"] == 1
        assert store.snapshot()["processed_critical_event_seq"] == 3
        assert store.snapshot()["pending_critical_events"] == []
        if act_before_refresh:
            return call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="old-plan")
        return call("memory_refresh")

    steps = [publish]
    if act_before_refresh:
        steps += [call("stop"), call("memory_refresh")]
    steps += [call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="unread-plan"),
              call("memory_read", {"path": path}),
              call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="fresh-plan"),
              call("finish")]
    metrics, events, backend, _ = run(tmp_path, Model(steps), store=store, read_only=True)
    results = tool_results(events)
    assert metrics["success"] and len(backend.moves) == 1
    assert any(r.get("error_code") == "MEMORY_REREAD_REQUIRED" for r in results)
    if act_before_refresh:
        assert results[0]["error_code"] == "MEMORY_CHANGED"
        assert results[1]["status"] == "succeeded"  # Recovery stop is always permitted.
    refresh = next(r for r in results if r["tool"] == "memory_refresh")
    assert refresh["required_reads"] == [path]
    assert metrics["memory_read_calls"] == 1


def test_critical_ack_without_document_change_requires_only_one_refresh(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    seed(store)

    def acknowledge(observation, history):
        notice = store.append_event("instruction", {"text": "No semantic change"}, critical=True)
        store.commit(1, notice["seq"], [], call_id="no-change", event_seqs=[notice["seq"]])
        return call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="unrefreshed")

    planner = Model([acknowledge, call("stop"), call("get_geometry"), call("memory_refresh"),
                     call("move_relative", {"forward_m": .2, "lateral_m": 0}, identifier="refreshed"),
                     call("finish")])
    metrics, events, backend, _ = run(tmp_path, planner, store=store, read_only=True)
    results = tool_results(events)
    assert metrics["success"] and len(backend.moves) == 1
    assert results[0]["error_code"] == "MEMORY_CHANGED"
    assert results[1]["status"] == results[2]["status"] == "succeeded"
    assert results[3]["required_reads"] == []
    assert metrics["memory_read_calls"] == 0


def test_selected_critical_version_does_not_make_unrelated_normal_updates_blocking(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {"text": "Current order"}, critical=True)
    store.commit(0, 1, [change(event)], call_id="initial-critical", event_seqs=[1])

    def ordinary_update(observation, history):
        event = store.append_event("observation", {"text": "A box"})
        store.commit(1, 2, [change(event, "A box", "objects/box.md")], call_id="ordinary")
        return call("move_relative", {"forward_m": .2, "lateral_m": 0})

    planner = Model([call("memory_read", {"path": "tasks/order.md"}), ordinary_update, call("finish")])
    metrics, events, backend, _ = run(tmp_path, planner, store=store, read_only=True)
    assert metrics["success"] and len(backend.moves) == 1
    assert all(r["status"] == "succeeded" for r in tool_results(events))


def test_pending_critical_notice_blocks_motion_but_allows_stop_and_abort(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    store.append_event("instruction", {"text": "Wait for changed order"}, critical=True)
    planner = Model([call("move_relative", {"forward_m": .2, "lateral_m": 0}),
                     call("stop"), call("abort", {"reason": "Awaiting memory"})])
    metrics, events, backend, _ = run(tmp_path, planner, store=store, read_only=True)
    assert metrics["termination"] == "abort" and not metrics["success"]
    assert backend.moves == []
    assert tool_results(events)[0]["error_code"] == "MEMORY_PENDING"


class Receipt:
    extension_id = "fixture-receipt"
    tools = [{"type": "function", "function": {"name": "submit_mission",
              "parameters": {"type": "object", "properties": {}, "required": []}}}]

    def __init__(self, success=True):
        self.calls = 0
        self.success = success

    def execute(self, call):
        self.calls += 1
        return {"call_id": call.call_id, "tool": call.name, "status": "succeeded"}

    def evaluate(self):
        return {"success": self.success and self.calls > 0, "submitted": self.calls > 0}


def test_receipt_requires_read_and_scoring_combines_physical_and_extension(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    seed(store)
    receipt = Receipt(success=False)
    planner = Model([call("submit_mission"), call("memory_read", {"path": "tasks/order.md"}),
                     call("submit_mission", identifier="read-receipt"), call("finish")])
    metrics, events, _, _ = run(tmp_path, planner, store=store, read_only=True, task_extension=receipt)
    assert receipt.calls == 1
    assert tool_results(events)[0]["error_code"] == "MEMORY_NOT_READ"
    assert metrics["evaluation"]["physical_success"] is True
    assert metrics["evaluation"]["task_extension"]["success"] is False
    assert metrics["success"] is False


def test_initial_memory_timeout_retains_notice_and_never_calls_planner(tmp_path):
    memory = Model([ModelError("SENSITIVE_BODY", "MODEL_TIMEOUT")])
    planner = Model([])
    started = time.monotonic()
    metrics, events, backend, store = run(tmp_path, planner, memory_model=memory,
                                        memory_notice="Remember this task", max_wall_seconds=30)
    assert time.monotonic() - started < 3
    assert metrics["termination"] == "memory_error" and metrics["model_calls"] == 0
    assert metrics["memory_agent"]["api_requests"] == 1
    assert metrics["memory_pending_events"] == 1
    assert store.pending_events()[0]["payload"]["text"] == "Remember this task"
    assert memory.closed == planner.closed == 1 and backend.stops == 1
    assert "SENSITIVE_BODY" not in json.dumps(metrics)


def test_planner_timeout_stops_then_flushes_pending_feedback_and_counts_both_models(tmp_path):
    planner = Model([call("observe"), ModelError("timeout", "MODEL_TIMEOUT")])
    memory = Model([call("commit_memory", {"changes": []})])
    metrics, events, backend, store = run(tmp_path, planner, memory_model=memory)
    assert metrics["termination"] == "model_error"
    assert metrics["api_requests_total"] == 3
    assert metrics["completed_responses"] == 1
    assert metrics["usage_total"]["total_tokens"] == 20
    assert not metrics["usage_complete"]
    assert not store.pending_events()
    assert backend.stops >= 1 and memory.closed == planner.closed == 1


def test_missing_usage_is_unknown_even_with_complete_responses(tmp_path):
    planner = Model([call("finish", usage={})])
    metrics, _, _, _ = run(tmp_path, planner, read_only=True)
    assert metrics["completed_responses"] == metrics["model_calls"] == 1
    assert metrics["usage_complete"] is False and metrics["usage_total"] == {}


def test_private_history_is_not_logged_and_only_public_feedback_is_registered(tmp_path):
    private = {"reasoning_content": "PRIVATE_REASONING_CANARY", "assistant_content": "PRIVATE_ASSISTANT_CANARY"}
    planner = Model([call("observe", **private), call("finish", **private)])
    memory = Model([call("commit_memory", {"changes": []})])
    metrics, events, _, store = run(tmp_path, planner, memory_model=memory)
    assert metrics["success"]
    assert planner.inputs[1][1][0]["reasoning_content"] == private["reasoning_content"]
    text = (tmp_path / "run" / "events.jsonl").read_text() + (tmp_path / "run" / "memory-agent.jsonl").read_text()
    assert all(value not in text for value in private.values())
    memory_input = json.dumps(memory.inputs)
    assert "hidden_target" not in memory_input and "base_xy_m" not in memory_input
    assert all(value not in memory_input for value in private.values())


def test_original_tracking_tools_do_not_require_any_memory_read(tmp_path):
    planner = Model([call("get_geometry"), call("finish")])
    metrics, events, _, _ = run(tmp_path, planner, read_only=True)
    assert metrics["success"]
    assert tool_results(events)[0]["status"] == "succeeded"
    assert metrics["memory_read_calls"] == 0


def test_navigation_rejects_memory_write_and_multiple_tools(tmp_path):
    planner = Model([call("commit_memory", {"changes": []}),
                     Decision([ToolCall("a", "move_relative", {"forward_m": .2, "lateral_m": 0}),
                               ToolCall("b", "finish", {})], usage=USAGE), call("finish")])
    metrics, events, backend, _ = run(tmp_path, planner, read_only=True)
    assert metrics["success"] and not backend.moves
    results = tool_results(events)
    assert results[0]["error_code"] == "UNKNOWN_TOOL"
    assert all(r["error_code"] == "ONE_TOOL_PER_TURN" for r in results[1:3])


def test_untransferred_memory_model_is_closed_when_setup_fails(tmp_path):
    class BrokenBackend(EpisodeBackend):
        def capture(self):
            raise RuntimeError("Unavailable camera")

    memory, planner = Model([]), Model([])
    metrics, _, backend, _ = run(tmp_path, planner, memory_model=memory, backend=BrokenBackend())
    assert not metrics["success"]
    assert memory.closed == planner.closed == 1
    assert backend.stops == 1 and not backend.closed


def test_shared_deadline_bounds_background_request_timeout(tmp_path):
    memory = Model([call("commit_memory", {"changes": []}), call("commit_memory", {"changes": []})])
    planner = Model([call("finish")])
    metrics, _, _, _ = run(tmp_path, planner, memory_model=memory, memory_notice="Order", max_wall_seconds=2,
                           api_timeout_s=180)
    assert metrics["success"]
    assert all(0 < entry[3] <= 2 for entry in memory.inputs)
    assert all(0 < entry[3] <= 2 for entry in planner.inputs)
