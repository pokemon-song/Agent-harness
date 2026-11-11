"""Real thread barriers and local HTTP fixtures; no paid model requests."""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from go2_demo.md_memory.store import MDMemoryStore
from go2_demo.md_memory.planner import MDPlanningAgent
from go2_demo.md_memory.worker import MemoryWorker, _EMPTY_PNG, memory_tools
from go2_demo.models import APIModel, ModelError
from go2_demo.robots.go2.profile import GO2_PROFILE
from go2_demo.types import Decision, ToolCall


def decision(name, arguments, call_id="call", **kwargs):
    return Decision([ToolCall(call_id, name, arguments)], **kwargs)


def change(event, body="The current task is to inspect the blue marker."):
    return {"path": "tasks/current.md", "body": body,
            "evidence_ids": [event["id"]], "status": "confirmed"}


class Scripted:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.inputs = []
        self.closed = 0

    def decide(self, observation, image_data_url, tools, history, *, timeout_s):
        self.inputs.append((observation, image_data_url, tools, history, timeout_s))
        step = next(self.steps)
        if isinstance(step, Exception):
            raise step
        return step(observation, image_data_url, tools, history) if callable(step) else step

    def close(self):
        self.closed += 1


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), "Worker did not reach the expected boundary"


def make_worker(tmp_path, store, model, **kwargs):
    return MemoryWorker(store, model, log_path=tmp_path / "worker.jsonl", **kwargs)


def logs(worker):
    return [json.loads(line) for line in worker.log_path.read_text().splitlines()]


def test_blocked_model_does_not_block_snapshot_or_concurrent_planning(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    first = store.append_event("instruction", {"text": "Inspect the blue marker"})
    store.commit(0, 1, [change(first)], call_id="seed")
    event = store.append_event("observation", {"text": "The marker remains visible"})
    entered, release = threading.Event(), threading.Event()

    def blocked(*_):
        entered.set()
        assert release.wait(3)
        return decision("commit_memory", {"changes": []})

    model = Scripted([blocked])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert entered.wait(2)

        def plan_from_published_memory():
            def propose(observation, image, tools, history):
                document = history[-1]["results"][0]["document"]
                return decision("submit_plan", {
                    "based_on_revision": observation["memory_snapshot"]["revision"],
                    "dependencies": [{"path": document["path"], "sha256": document["sha256"]}],
                    "subgoal": document["body"], "completion_condition": "New observation is available",
                    "on_failure": "stop_and_replan", "action_name": "observe", "arguments": {}})

            # Exercise the real independent planner and its read receipts while
            # the other model is demonstrably still inside its blocking call.
            planning_model = Scripted([decision("memory_read", {"path": "tasks/current.md"}), propose])
            harness = SimpleNamespace(profile=GO2_PROFILE, observation=lambda: {"robot_id": "go2"}, tools=[])
            planner = MDPlanningAgent(store, planning_model, harness, log_path=tmp_path / "planner.jsonl")
            try:
                return planner.plan("Inspect the marker", "image", timeout_s=1)
            finally:
                planner.close()

        with ThreadPoolExecutor(max_workers=1) as pool:
            plan = pool.submit(plan_from_published_memory).result(timeout=1)
        assert plan.snapshot_revision == 1
        assert plan.subgoal.endswith("blue marker.")
        assert store.snapshot()["processed_event_seq"] == 1
        release.set()
        assert worker.wait_until(event["seq"], 2)
    finally:
        release.set()
        assert worker.stop(2)
    assert model.closed == 1


def test_transport_timeout_preserves_pending_until_explicit_notify(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {"text": "Inspect the marker"})
    requests = []

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            await asyncio.sleep(.15)
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [{"id": "fixture-call", "type": "function", "function": {
                "name": "commit_memory", "arguments": json.dumps({"changes": [change(event)]})}}]}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}})

    model = APIModel("https://example.invalid/v1", "fixture", "fixture-key", transport=httpx.MockTransport(handler))
    worker = make_worker(tmp_path, store, model, request_timeout_s=.04, max_calls=3).start()
    try:
        wait_for(lambda: bool(worker.stats["errors"]))
        assert worker.stats["errors"][0]["code"] == "MODEL_TIMEOUT"
        error_log = next(entry for entry in logs(worker) if entry["event"] == "model_error")
        assert error_log["attempt"] == 1 and error_log["latency_s"] >= .03
        assert error_log["request_timeout_s"] == .04
        assert not worker.wait_until(event["seq"], .08)
        assert len(requests) == 1
        assert store.pending_events() == [event]
        worker.notify()
        assert worker.wait_until(event["seq"], 2)
        assert len(requests) == 2
        assert worker.stats["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    finally:
        assert worker.stop(2)


def test_new_worker_resumes_durable_pending_after_failure(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {"text": "Return"})
    failed = Scripted([ModelError("Secret transport detail", "MODEL_TIMEOUT")])
    worker = make_worker(tmp_path, store, failed).start()
    wait_for(lambda: bool(worker.stats["errors"]))
    assert worker.stop(2)
    assert failed.closed == 1 and store.pending_events() == [event]
    resumed_model = Scripted([decision("commit_memory", {"changes": [change(event)]})])
    resumed = make_worker(tmp_path, MDMemoryStore(store.root), resumed_model).start()
    try:
        assert resumed.wait_until(event["seq"], 2)
    finally:
        assert resumed.stop(2)
    assert "Secret transport detail" not in worker.log_path.read_text()


def test_late_old_base_cannot_overwrite_concurrent_publication(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {"text": "New instruction"})
    entered, release = threading.Event(), threading.Event()

    def old_reply(*_):
        entered.set()
        assert release.wait(3)
        return decision("commit_memory", {"changes": [change(event, "Old late content")]})

    worker = make_worker(tmp_path, store, Scripted([old_reply])).start()
    try:
        assert entered.wait(2)
        store.commit(0, 1, [change(event, "New accepted content")], call_id="other-writer")
        release.set()
        wait_for(lambda: bool(worker.stats["errors"]))
        assert worker.stats["errors"][0]["code"] == "MEMORY_VERSION_CONFLICT"
        assert store.read("tasks/current.md")["body"] == "New accepted content"
        assert worker.stats["api_requests"] == 1
        assert worker.stats["committed_batches"] == 0
    finally:
        release.set()
        assert worker.stop(2)


def test_no_robot_tools_and_rejected_tool_can_be_corrected(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("observation", {})
    model = Scripted([decision("move_relative", {"forward_m": .2, "lateral_m": 0}),
                      decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(event["seq"], 2)
    finally:
        assert worker.stop(2)
    assert {tool["function"]["name"] for tool in memory_tools()} == {
        "memory_search", "memory_read", "inspect_event", "commit_memory", "patch_memory"}
    assert model.inputs[1][3][0]["results"][0]["error_code"] == "UNSUPPORTED_TOOL"
    assert worker.stats["tool_errors"] == 1
    assert store.snapshot()["files"] == []


def test_stop_is_bounded_and_discards_inflight_response_before_close(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("observation", {})
    entered, release = threading.Event(), threading.Event()

    def blocked(*_):
        entered.set()
        assert release.wait(3)
        return decision("commit_memory", {"changes": [change(event)]})

    model = Scripted([blocked])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert entered.wait(2)
        started = time.monotonic()
        assert worker.stop(.02) is False
        assert time.monotonic() - started < .3
        assert model.closed == 0
        release.set()
        assert worker.stop(2)
        assert worker.stop(0)
    finally:
        release.set()
        worker.stop(2)
    assert model.closed == 1
    assert store.pending_events() == [event]
    assert any(e["event"] == "discarded_response" for e in logs(worker))


def test_private_provider_fields_stay_only_in_bounded_history(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {"text": "Blue marker"})
    private = {"reasoning_content": "PRIVATE_REASONING_SENTINEL",
               "assistant_content": "PRIVATE_ASSISTANT_SENTINEL", "raw_text": "PRIVATE_RAW_SENTINEL"}
    model = Scripted([
        decision("memory_search", {"query": ""}, **private),
        decision("memory_search", {"query": "blue"}, **private),
        decision("memory_search", {"query": "marker"}, **private),
        decision("commit_memory", {"changes": [change(event)]}, **private),
    ])
    worker = make_worker(tmp_path, store, model, history_turns=2).start()
    try:
        assert worker.wait_until(event["seq"], 2)
    finally:
        assert worker.stop(2)
    assert [len(item[3]) for item in model.inputs] == [0, 1, 2, 2]
    assert model.inputs[-1][3][-1]["reasoning_content"] == private["reasoning_content"]
    text = worker.log_path.read_text()
    assert all(value not in text for value in private.values())
    assert all(value not in store.read("tasks/current.md")["content"] for value in private.values())


def test_inspect_registered_image_and_memory_reads_use_pinned_revision(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    (store.root / "evidence").mkdir()
    (store.root / "evidence" / "first.png").write_bytes(base64.b64decode(_EMPTY_PNG))
    first = store.append_event("observation", {"text": "First image"}, image_path="evidence/first.png")
    store.commit(0, 1, [change(first)], call_id="seed")
    event = store.append_event("observation", {"text": "Review earlier evidence"})
    model = Scripted([decision("memory_read", {"path": "tasks/current.md"}),
                      decision("inspect_event", {"event_id": first["id"]}),
                      decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(event["seq"], 2)
    finally:
        assert worker.stop(2)
    assert model.inputs[0][0]["current_image"]["available"] is False
    assert model.inputs[2][0]["current_image"]["event_id"] == first["id"]
    assert model.inputs[2][1] == "data:image/png;base64," + _EMPTY_PNG
    assert model.inputs[1][3][0]["results"][0]["document"]["snapshot_revision"] == 1


def test_multi_tool_response_is_rejected_without_committing(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("observation", {})
    model = Scripted([Decision([ToolCall("a", "commit_memory", {"changes": []}),
                                ToolCall("b", "memory_search", {"query": ""})]),
                      decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(event["seq"], 2)
    finally:
        assert worker.stop(2)
    assert all(r["error_code"] == "ONE_TOOL_PER_TURN" for r in model.inputs[1][3][0]["results"])
    assert store.snapshot()["revision"] == 1


def test_batches_are_at_most_eight_and_budget_is_global(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    events = [store.append_event("observation", {"index": index}) for index in range(17)]
    model = Scripted([decision("commit_memory", {"changes": []}) for _ in range(2)])
    worker = make_worker(tmp_path, store, model, max_calls=2).start()
    wait_for(lambda: worker.stats["stopped"])
    assert worker.stop(2)
    assert [len(item[0]["pending_events"]) for item in model.inputs] == [8, 8]
    assert [item[0]["batch_last_seq"] for item in model.inputs] == [8, 16]
    assert store.pending_events() == [events[-1]]
    assert worker.stats["api_requests"] == 2
    assert model.closed == 1


def test_factory_is_owned_and_created_on_worker_thread(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("observation", {})
    creator = []
    model = Scripted([decision("commit_memory", {"changes": []})])

    def factory():
        creator.append(threading.get_ident())
        return model

    worker = make_worker(tmp_path, store, factory)
    assert creator == []
    worker.start()
    try:
        assert worker.wait_until(event["seq"], 2)
    finally:
        assert worker.stop(2)
    assert creator != [threading.get_ident()]
    assert len(creator) == 1 and model.closed == 1


def test_unknown_evidence_cannot_be_published_and_can_be_corrected(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {"text": "Blue marker"})
    forged = change({"id": "not-a-registered-event"})
    model = Scripted([decision("commit_memory", {"changes": [forged]}),
                      decision("commit_memory", {"changes": [change(event)]})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert worker.wait_until(event["seq"], 2)
    finally:
        assert worker.stop(2)
    assert worker.stats["tool_errors"] == 1
    assert store.read("tasks/current.md")["evidence_ids"] == [event["id"]]


def test_new_event_cannot_be_used_to_acknowledge_beyond_fixed_batch(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    first = store.append_event("instruction", {"text": "Old instruction"})
    later = []

    def reply_with_future_evidence(*_):
        later.append(store.append_event("instruction", {"text": "New instruction"}, critical=True))
        return decision("commit_memory", {"changes": [change(later[0])]})

    model = Scripted([reply_with_future_evidence,
                      decision("commit_memory", {"changes": []}),
                      decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        wait_for(lambda: bool(later))
        assert worker.wait_until(later[0]["seq"], 2)
    finally:
        assert worker.stop(2)
    assert [item[0]["batch_last_seq"] for item in model.inputs] == [first["seq"], later[0]["seq"], first["seq"]]
    assert any(row.get("reason") == "new_critical_event" for row in logs(worker))
    assert store.snapshot()["files"] == []


def test_notification_before_failure_does_not_become_automatic_retry(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("instruction", {})
    entered, release = threading.Event(), threading.Event()

    def fail_after_notification(*_):
        entered.set()
        assert release.wait(3)
        raise ModelError("Timed out", "MODEL_TIMEOUT")

    model = Scripted([fail_after_notification, decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model).start()
    try:
        assert entered.wait(2)
        worker.notify()
        release.set()
        wait_for(lambda: bool(worker.stats["errors"]))
        assert not worker.wait_until(event["seq"], .06)
        assert worker.stats["api_requests"] == 1
        worker.notify()
        assert worker.wait_until(event["seq"], 2)
    finally:
        release.set()
        assert worker.stop(2)


def test_success_at_exact_call_budget_is_not_reported_as_failure(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("observation", {})
    model = Scripted([decision("commit_memory", {"changes": []})])
    worker = make_worker(tmp_path, store, model, max_calls=1).start()
    try:
        assert worker.wait_until(event["seq"], 2)
        assert worker.stats["errors"] == []
    finally:
        assert worker.stop(2)
    assert worker.stats["api_requests"] == 1
    assert worker.stats["errors"] == []


def test_notify_immediately_after_published_failure_is_not_lost(tmp_path):
    store = MDMemoryStore(tmp_path / "memory")
    event = store.append_event("observation", {})
    published, resume = threading.Event(), threading.Event()

    class PausedErrorWorker(MemoryWorker):
        def _error(self, exc, **fields):
            super()._error(exc, **fields)
            published.set()
            assert resume.wait(3)

    model = Scripted([ModelError("timeout", "MODEL_TIMEOUT"),
                      decision("commit_memory", {"changes": []})])
    worker = PausedErrorWorker(store, model, log_path=tmp_path / "worker.jsonl").start()
    try:
        assert published.wait(2)
        assert worker.stats["errors"]
        worker.notify()
        resume.set()
        assert worker.wait_until(event["seq"], 2)
    finally:
        resume.set()
        assert worker.stop(2)
    assert worker.stats["api_requests"] == 2
