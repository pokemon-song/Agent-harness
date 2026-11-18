"""Paired return-loop integration at the public harness, without model or physics services."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from go2_demo.harness import Harness
from go2_demo.occluded_recall import Recall
from go2_demo.occluded_return import run_return
from go2_demo.runner import RunLog
from go2_demo.types import Decision, ToolCall
from test_runner import EpisodeBackend


EXCERPT = "B was seen beside the cabinet in the outbound camera."
DOCS = {"route.md": "# Route\n[E001] " + EXCERPT}
COUNTERS = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
MOVE = ("move_relative", {"forward_m": .2, "lateral_m": 0.})
FINISH = ("finish", {})


class PublicBackend(EpisodeBackend):
    _fallen_ever = False
    collision_terminated = False

    def evaluate(self):
        return super().evaluate(standoff_m=1., tolerance_m=.18)


class Planner:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.inputs = []

    def decide(self, observation, image, tools, history, **kwargs):
        self.inputs.append((deepcopy(observation), deepcopy(history)))
        assert image.startswith("data:image/") and kwargs["timeout_s"] > 0
        name, args = next(self.actions)
        assert name in {tool["function"]["name"] for tool in tools}
        self.raw_usage = COUNTERS
        return Decision([ToolCall(str(len(self.inputs)), name, args)])


class MemoryResponder:
    received_response = True
    last_usage = COUNTERS

    def select(self, question, documents, *, timeout_s):
        assert documents == DOCS and question == "Where was B?"
        return {"support": "partial", "ids": ["E001"]}


def read_action(mode):
    if mode == "direct":
        return "memory_read", {"path": "route.md"}
    return "memory_query", {"question": "Where was B?"}


def exercise(tmp_path, mode, actions, budget, backend=None):
    backend = backend or PublicBackend()
    harness = Harness(backend, "fixture task")
    recall = Recall(DOCS, mode, MemoryResponder() if mode == "query" else None)
    planner = Planner(actions)
    args = SimpleNamespace(max_wall_seconds=100, max_sim_seconds=120, max_requests=budget, api_timeout=30)
    log = RunLog(tmp_path)
    try:
        metrics = run_return(args, backend, harness, planner, recall, log)
    finally:
        log.close()
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    results = [result for e in events if e["event"] == "tool_results" for result in e["results"]]
    return metrics, planner.inputs, backend, results


@pytest.mark.parametrize("mode", ["direct", "query"])
def test_read_gate_then_real_dispatch_and_exactly_one_current_copy(tmp_path, mode):
    actions = [FINISH, MOVE, read_action(mode), MOVE, FINISH]
    metrics, inputs, backend, results = exercise(tmp_path, mode, actions, 6)
    assert [item["error_code"] for item in results[:2]] == ["OUTBOUND_MEMORY_REQUIRED"] * 2
    assert len(backend.moves) == 1 and backend.moves[0] == MOVE[1]
    assert metrics["success"] and metrics["termination"] == "finish"
    for obs, history in inputs[3:]:
        assert json.dumps(obs).count(EXCERPT) == 1
        assert EXCERPT not in json.dumps(history)
        assert all("return_memory" not in turn["observation"] for turn in history)
    assert EXCERPT not in json.dumps(results[2])
    assert metrics["planner"]["requests"] == 5
    assert metrics["memory_reader"]["requests"] == int(mode == "query")
    assert metrics["total_requests"] == 5 + int(mode == "query")
    assert metrics["known_usage"]["total_tokens"] == 12 * metrics["total_requests"]


@pytest.mark.parametrize("mode", ["direct", "query"])
def test_last_shared_request_slot_only_allows_local_retrieval(tmp_path, mode):
    metrics, _, _, results = exercise(tmp_path, mode, [read_action(mode)], 1)
    assert metrics["total_requests"] == 1 and metrics["memory_reader"]["requests"] == 0
    assert metrics["termination"] == "budget_exhausted" and not metrics["success"]
    if mode == "direct":
        assert results[0]["status"] == "succeeded"
    else:
        assert results[0]["error_code"] == "MODEL_REQUEST_BUDGET"


def test_first_contact_during_finish_wins_over_a_successful_evaluator(tmp_path):
    class ContactOnStop(PublicBackend):
        def stop(self):
            self.collision_terminated = True
            return super().stop()

    metrics, inputs, _, _ = exercise(tmp_path, "direct", [read_action("direct"), FINISH], 8, ContactOnStop())
    assert len(inputs) == 2 and metrics["evaluation"]["success"]
    assert not metrics["success"] and metrics["termination"] == "collision"
