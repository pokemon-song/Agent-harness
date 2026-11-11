"""Read-only planning loop and a short-action gate; no automatic plan generation."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
import uuid

from ..harness import _tool
from ..models import ModelError
from ..types import ToolCall


class PlanError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ShortPlan:
    plan_id: str
    snapshot_revision: int
    dependencies: tuple
    action_name: str
    arguments: dict
    subgoal: str
    completion_condition: str
    scope: str
    expires_monotonic: float = field(repr=False)
    on_failure: str = "stop_and_replan"

    def public(self):
        result = asdict(self)
        result.pop("expires_monotonic")
        return result


def check_plan(plan, store, profile):
    """Check again at each synchronous action boundary. Does not assess semantics."""
    if time.monotonic() > plan.expires_monotonic:
        raise PlanError("PLAN_EXPIRED", "Plan dispatch deadline expired")
    snapshot = store.snapshot()
    if snapshot["scope"] != plan.scope:
        raise PlanError("SCOPE_CHANGED", "Plan belongs to a different world scope")
    if snapshot["pending_critical_events"]:
        raise PlanError("MEMORY_PENDING", "Critical inputs have not been processed")
    if snapshot.get("processed_critical_event_seq", 0) > store.snapshot(
            plan.snapshot_revision).get("processed_critical_event_seq", 0):
        raise PlanError("MEMORY_CHANGED", "A new critical input was published after this plan snapshot")
    if not plan.dependencies:
        raise PlanError("NO_MEMORY_DEPENDENCY", "This memory demo requires read evidence")
    for dependency in plan.dependencies:
        current = store.read(dependency["path"], revision=snapshot["revision"])
        if current["sha256"] != dependency["sha256"]:
            raise PlanError("MEMORY_CHANGED", "A plan dependency was revised")
        if current.get("status") in {"unknown", "stale", "contradicted"}:
            raise PlanError("MEMORY_UNCERTAIN", "A required memory is not usable")
        if current.get("status") == "predicted" and plan.action_name not in {"observe", "stop"}:
            raise PlanError("MEMORY_UNCERTAIN", "Predicted memory requires observation first")
    if plan.action_name not in {"move_relative", "turn", "stop", "observe"}:
        raise PlanError("ACTION_NOT_ALLOWED", "Only short demo actions are supported")
    try:
        profile.validate_action(plan.action_name, plan.arguments)
        if plan.action_name == "move_relative" and math.hypot(
                plan.arguments["forward_m"], plan.arguments["lateral_m"]) > .30 + 1e-9:
            raise ValueError("Demo translation limit is 0.30 metres")
    except (ValueError, TypeError, KeyError) as exc:
        raise PlanError("INVALID_ACTION", "Action is outside the robot/demo capability limits") from exc
    return snapshot


def execute_plan(plan, store, harness):
    check_plan(plan, store, harness.profile)
    # Existing Harness caches this ID, so repeating dispatch in this process
    # cannot perform the same physical action twice. Cross-process recovery must
    # reconcile execution feedback; it must not blindly replay an action.
    return harness.execute(ToolCall(plan.plan_id, plan.action_name, deepcopy(plan.arguments)))


class MDPlanningAgent:
    def __init__(self, store, model, harness, *, log_path, max_calls=8,
                 request_timeout_s=90, history_turns=2):
        if (type(max_calls) is not int or max_calls < 1
                or isinstance(request_timeout_s, bool) or not math.isfinite(request_timeout_s)
                or request_timeout_s <= 0 or type(history_turns) is not int or not 0 <= history_turns <= 100):
            raise ValueError("Positive planning budgets required")
        self.store, self.model, self.harness = store, model, harness
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.timeout_s, self.history_turns = max_calls, request_timeout_s, history_turns
        self.model.system_prompt = (Path(__file__).parents[1] / "prompts/md_planner.txt").read_text()
        self.stats = {"api_requests": 0, "completed_responses": 0, "usage": {}, "errors": []}
        self._closed = False

    def _log(self, event, **data):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, **data}, ensure_ascii=False, allow_nan=False) + "\n")

    def _tools(self):
        string = {"type": "string"}
        return [
            _tool("memory_search", "Search this published memory snapshot.", {"query": string}, ["query"]),
            _tool("memory_read", "Read one complete published Markdown document and its evidence references.",
                  {"path": string}, ["path"]),
            _tool("memory_refresh", "Select the latest complete snapshot; prior read receipts are cleared."),
            _tool("abort_plan", "Decline to guess missing instructions or act on uncertain evidence.",
                  {"reason": string}, ["reason"]),
            _tool("submit_plan", "Propose one bounded robot action; this tool never executes it.", {
                "based_on_revision": {"type": "integer", "minimum": 0},
                "dependencies": {"type": "array", "minItems": 1, "maxItems": 6, "items": {
                    "type": "object", "properties": {"path": string, "sha256": string},
                    "required": ["path", "sha256"], "additionalProperties": False}},
                "subgoal": string, "completion_condition": string,
                "on_failure": {"type": "string", "enum": ["stop_and_replan"]},
                "action_name": {"type": "string", "enum": ["move_relative", "turn", "stop", "observe"]},
                "arguments": {"type": "object", "properties": {
                    "forward_m": {"type": "number"}, "lateral_m": {"type": "number"},
                    "yaw_rad": {"type": "number"}}, "additionalProperties": False},
            }, ["based_on_revision", "dependencies", "subgoal", "completion_condition", "on_failure", "action_name", "arguments"]),
        ]

    def _accept(self, args, snapshot, receipts):
        fields = {"based_on_revision", "dependencies", "subgoal", "completion_condition",
                  "on_failure", "action_name", "arguments"}
        if not isinstance(args, dict) or set(args) != fields:
            raise PlanError("INVALID_PLAN", "Unexpected plan fields")
        if type(args["based_on_revision"]) is not int or args["based_on_revision"] != snapshot["revision"]:
            raise PlanError("SNAPSHOT_MISMATCH", "Use the current planning snapshot")
        if any(not isinstance(args[k], str) or not 1 <= len(args[k]) <= 1000
               for k in ("subgoal", "completion_condition")) or args["on_failure"] != "stop_and_replan":
            raise PlanError("INVALID_PLAN", "Bounded descriptions and a recovery branch are required")
        dependencies = args["dependencies"]
        if not isinstance(dependencies, list) or not 1 <= len(dependencies) <= 6:
            raise PlanError("INVALID_PLAN", "Read dependencies are required")
        for dependency in dependencies:
            if (not isinstance(dependency, dict) or set(dependency) != {"path", "sha256"}
                    or not isinstance(dependency["path"], str)
                    or receipts.get(dependency["path"]) != dependency["sha256"]):
                raise PlanError("UNREAD_MEMORY", "Cite hashes returned by memory_read")
        plan = ShortPlan("plan-" + uuid.uuid4().hex, snapshot["revision"], tuple(deepcopy(dependencies)),
                         args["action_name"], deepcopy(args["arguments"]), args["subgoal"],
                         args["completion_condition"], snapshot["scope"], time.monotonic() + 30.)
        check_plan(plan, self.store, self.harness.profile)
        return plan

    def plan(self, task, image_data_url, *, timeout_s=180):
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("A finite positive planning timeout is required")
        if self._closed:
            raise PlanError("PLANNER_CLOSED", "Create a new planning agent after closing")
        deadline = time.monotonic() + timeout_s
        snapshot, receipts, history = self.store.snapshot(), {}, []
        for _ in range(max(0, self.max_calls - self.stats["api_requests"])):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PlanError("PLANNER_DEADLINE", "Planning wall budget exhausted")
            observation = {"task": task, "memory_snapshot": snapshot,
                           "observation": self.harness.observation(),
                           "robot_tools": self.harness.tools,
                           "remaining_calls": self.max_calls - self.stats["api_requests"]}
            self.stats["api_requests"] += 1
            self._log("model_input", attempt=self.stats["api_requests"], observation=observation)
            started = time.monotonic()
            try:
                decision = self.model.decide(observation, image_data_url, self._tools(), history,
                                             timeout_s=min(remaining, self.timeout_s))
            except Exception as exc:
                error = {"type": type(exc).__name__, "code": getattr(exc, "code", "MODEL_ERROR")}
                self.stats["errors"].append(error)
                self._log("model_error", **error, latency_s=time.monotonic()-started)
                raise
            self.stats["completed_responses"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = decision.usage.get(key)
                if type(value) is int and value >= 0:
                    self.stats["usage"][key] = self.stats["usage"].get(key, 0) + value
            calls = [asdict(c) for c in decision.calls]
            self._log("model_response", calls=calls, usage=decision.usage, latency_s=time.monotonic()-started)
            if time.monotonic() >= deadline:
                raise PlanError("PLANNER_DEADLINE", "Late plan response was discarded")
            results, accepted = [], None
            for call in decision.calls:
                result = {"call_id": call.call_id, "tool": call.name, "status": "succeeded"}
                try:
                    if len(decision.calls) != 1:
                        raise PlanError("ONE_TOOL_PER_TURN", "Use exactly one tool per response")
                    if call.name == "memory_search":
                        if set(call.arguments) != {"query"}:
                            raise PlanError("INVALID_ARGUMENTS", "Search takes query only")
                        result["matches"] = self.store.search(call.arguments["query"], revision=snapshot["revision"])
                    elif call.name == "memory_read":
                        if set(call.arguments) != {"path"}:
                            raise PlanError("INVALID_ARGUMENTS", "Read takes path only")
                        item = self.store.read(call.arguments["path"], revision=snapshot["revision"])
                        receipts[item["path"]] = item["sha256"]
                        result["document"] = item
                    elif call.name == "memory_refresh":
                        if call.arguments:
                            raise PlanError("INVALID_ARGUMENTS", "Refresh takes no arguments")
                        snapshot, receipts = self.store.snapshot(), {}
                        result["snapshot"] = snapshot
                    elif call.name == "submit_plan":
                        accepted = self._accept(call.arguments, snapshot, receipts)
                        result["plan"] = accepted.public()
                    elif call.name == "abort_plan":
                        raise PlanError("PLANNER_ABORT", "Planner declined to act")
                    else:
                        raise PlanError("UNKNOWN_TOOL", "Planner cannot write memory or execute robot tools")
                except Exception as exc:
                    if getattr(exc, "code", None) == "PLANNER_ABORT":
                        self._log("planner_abort")
                        raise
                    result.update(status="failed", error_code=getattr(exc, "code", "INVALID_ARGUMENTS"))
                results.append(result)
            self._log("tool_results", results=results)
            history.append({"observation": observation, "calls": calls, "results": results,
                            "reasoning_content": decision.reasoning_content,
                            "assistant_content": decision.assistant_content})
            history = history[-self.history_turns:] if self.history_turns else []
            if accepted:
                return accepted
        raise PlanError("PLANNER_BUDGET", "Planner call budget exhausted")

    def close(self):
        if not self._closed:
            self._closed = True
            self.model.close()
