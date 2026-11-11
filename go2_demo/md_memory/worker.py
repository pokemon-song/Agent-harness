"""Independent multimodal memory loop; no robot/controller access.

The model chooses Markdown changes. The host only bounds requests, pins reads to
a published revision, and acknowledges exactly the host-selected event set. A failed batch is
left pending until an explicit notification (or another worker) retries it.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid

from ..models import ModelError, ModelProtocolError
from .store import StoreError
from .contracts import MemoryToolError, apply_patch_edits, memory_tools, validate_tool
from .context import PROJECTION_VERSION, document_context, document_index, encoded_bytes, project_event


_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "md_memory_agent.txt"
_EMPTY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWZkAAAAASUVORK5CYII="
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _safe_error(exc, fallback="WORKER_ERROR"):
    code = getattr(exc, "code", fallback)
    if not isinstance(code, str) or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", code) is None:
        code = fallback
    return {"type": type(exc).__name__[:64], "code": code}


class MemoryWorker:
    """One owned model, one background thread, and a total model-call budget.

    ``model`` is an exclusive model instance or a zero-argument factory. Factories
    run inside the worker. ``stop`` never closes a model still making a request:
    it returns False if its bounded join expires; that worker closes its own model
    exactly once when the in-flight request returns. A stopped worker cannot be
    restarted; use a new instance to resume the durable pending events.
    """

    def __init__(self, store, model, *, log_path, max_calls=12,
                 request_timeout_s=90, history_turns=2):
        if type(max_calls) is not int or not 1 <= max_calls <= 10000:
            raise ValueError("max_calls must be an integer in 1..10000")
        if (isinstance(request_timeout_s, bool) or not isinstance(request_timeout_s, (int, float))
                or not math.isfinite(request_timeout_s) or request_timeout_s <= 0):
            raise ValueError("request_timeout_s must be finite and positive")
        if type(history_turns) is not int or not 0 <= history_turns <= 100:
            raise ValueError("history_turns must be an integer in 0..100")
        self.store, self._model_source = store, model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.request_timeout_s = float(request_timeout_s)
        self.history_turns = history_turns
        self._condition = threading.Condition()
        self._thread = None
        self._stop_requested = False
        self._wake = 1
        self._handled_wake = 0
        self._model = None
        self._id = uuid.uuid4().hex
        self._started_at = time.monotonic()
        self._stats = {"api_requests": 0, "completed_responses": 0, "usage": {},
                       "errors": [], "superseded_errors": [], "tool_errors": 0, "committed_batches": 0,
                       "preempted_batches": 0, "event_bytes_original": 0, "event_bytes_projected": 0,
                       "processed_event_seq": 0, "running": False, "stopped": False,
                       "max_calls": max_calls}

    @property
    def stats(self):
        with self._condition:
            return deepcopy(self._stats)

    def start(self):
        with self._condition:
            if self._thread is not None:
                return self
            if self._stop_requested:
                raise RuntimeError("A stopped memory worker cannot start")
            self._thread = threading.Thread(target=self._run, name="md-memory-" + self._id[:8], daemon=True)
            self._stats["running"] = True
            self._thread.start()
        return self

    def notify(self):
        with self._condition:
            if self._stop_requested or self._stats["stopped"]:
                return False
            self._wake += 1
            self._condition.notify_all()
        return True

    @staticmethod
    def _duration(value):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError("timeout_s must be finite and nonnegative")
        return float(value)

    def wait_until(self, seq, timeout_s):
        """Wait for the contiguous prefix; urgent acknowledgements do not skip gaps."""
        return self._wait(seq, timeout_s, individual=False)

    def wait_event(self, seq, timeout_s):
        """Wait for one registered event, independently of older normal feedback."""
        return self._wait(seq, timeout_s, individual=True)

    def _wait(self, seq, timeout_s, *, individual):
        if type(seq) is not int or seq < 0:
            raise ValueError("seq must be a nonnegative integer")
        deadline = time.monotonic() + self._duration(timeout_s)
        while True:
            # Do not hold the worker lock during any store operation.
            if (self.store.is_event_processed(seq) if individual
                    else self.store.snapshot()["processed_event_seq"] >= seq):
                return True
            with self._condition:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stats["stopped"]:
                    return False
                self._condition.wait(timeout=min(remaining, .25))

    def stop(self, timeout_s=5):
        timeout_s = self._duration(timeout_s)
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
            thread = self._thread
            if thread is None:
                self._stats["stopped"] = True
                return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout_s)
        return not thread.is_alive()

    def _stopping(self):
        with self._condition:
            return self._stop_requested

    def _log(self, event, **fields):
        record = {"event": event, "worker_id": self._id,
                  "elapsed_s": time.monotonic() - self._started_at, **fields}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _error(self, exc, **fields):
        error = _safe_error(exc)
        with self._condition:
            self._stats["errors"].append(error)
            self._condition.notify_all()
        # Exception text may contain transport secrets or model content.
        self._log("failure", error=error, **fields)

    def _run(self):
        try:
            self._model = (self._model_source if hasattr(self._model_source, "decide")
                           else self._model_source())
            self._model.system_prompt = _PROMPT.read_text(encoding="utf-8")
            processed = self.store.snapshot()["processed_event_seq"]
            with self._condition:
                self._stats["processed_event_seq"] = processed
            while not self._stopping():
                with self._condition:
                    while self._wake <= self._handled_wake and not self._stop_requested:
                        self._condition.wait()
                    if self._stop_requested:
                        break
                    self._handled_wake = self._wake
                while not self._stopping():
                    snapshot = self.store.snapshot()
                    events = self.store.pending_events(limit=8, prioritize_critical=True)
                    if not events:
                        break
                    if self.stats["api_requests"] >= self.max_calls:
                        self._error(ModelError("Worker call budget exhausted", "MEMORY_CALL_BUDGET"))
                        return
                    try:
                        committed = self._batch(snapshot, events)
                    except Exception as exc:
                        # A notification received before this failure is not an
                        # implicit retry. Consume it before publishing the error,
                        # so a caller reacting to that error cannot lose its wake.
                        with self._condition:
                            self._handled_wake = self._wake
                        self._error(exc, base_revision=snapshot["revision"], last_seq=events[-1]["seq"])
                        break
                    if not committed:
                        break
        except Exception as exc:
            self._error(exc)
        finally:
            if self._model is not None:
                try:
                    self._model.close()
                except Exception as exc:
                    self._error(exc)
            with self._condition:
                self._stats["running"] = False
                self._stats["stopped"] = True
                self._condition.notify_all()

    def _image(self, event):
        if event is None or not event.get("image_path"):
            return "data:image/png;base64," + _EMPTY_PNG, {"available": False, "placeholder": True}
        relative = Path(event["image_path"])
        root = self.store.root.resolve()
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ModelError("Invalid evidence image", "INVALID_EVENT_IMAGE")
        if path.stat().st_size > 5_500_000:
            raise ModelError("Evidence image too large", "INVALID_EVENT_IMAGE")
        body = path.read_bytes()
        if not body.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ModelError("Evidence image must be PNG", "INVALID_EVENT_IMAGE")
        return "data:image/png;base64," + base64.b64encode(body).decode("ascii"), {
            "available": True, "event_id": event["id"], "path": event["image_path"],
            "sha256": hashlib.sha256(body).hexdigest()}

    def _batch(self, snapshot, events):
        base = snapshot["revision"]
        event_seqs = [event["seq"] for event in events]
        last_seq = max(event_seqs)
        if (event_seqs != sorted(set(event_seqs)) or self.store.snapshot()["revision"] != base
                or any(self.store.is_event_processed(seq) for seq in event_seqs)):
            raise ModelError("Batch snapshot changed", "MEMORY_SNAPSHOT_CHANGED")
        batch_id = uuid.uuid4().hex
        image_url, image_meta = self._image(next((e for e in reversed(events) if e.get("image_path")), None))
        history = []
        reads = set()
        projected = [project_event(event) for event in events]
        original_bytes, projected_bytes = encoded_bytes(events), encoded_bytes(projected)
        tools = memory_tools()
        while not self._stopping():
            if self._preempt_for_critical(events, batch_id):
                return True
            with self._condition:
                if self._stop_requested:
                    return False
                if self._stats["api_requests"] >= self.max_calls:
                    raise ModelError("Worker call budget exhausted", "MEMORY_CALL_BUDGET")
                self._stats["api_requests"] += 1
                attempt = self._stats["api_requests"]
            observation = {"role": "memory_agent", "scope": snapshot.get("scope"),
                           "base_revision": base, "batch_last_seq": last_seq,
                           "batch_event_seqs": event_seqs,
                           "batch_priority": "critical" if any(e["critical"] for e in events) else "normal",
                           "pending_events": deepcopy(projected), "memory_index": document_index(snapshot["files"]),
                           "current_image": image_meta,
                           "remaining_model_calls": self.max_calls - attempt + 1}
            self._log("model_request", attempt=attempt, batch_id=batch_id, base_revision=base,
                      event_ids=[e["id"] for e in events], last_seq=last_seq,
                      event_seqs=event_seqs, batch_priority=observation["batch_priority"],
                      projection=PROJECTION_VERSION, event_bytes_original=original_bytes,
                      event_bytes_projected=projected_bytes,
                      request_timeout_s=self.request_timeout_s, image=image_meta)
            with self._condition:
                self._stats["event_bytes_original"] += original_bytes
                self._stats["event_bytes_projected"] += projected_bytes
            started = time.monotonic()
            try:
                decision = self._model.decide(deepcopy(observation), image_url, deepcopy(tools),
                                              deepcopy(history), timeout_s=self.request_timeout_s)
            except Exception as exc:
                self._log("model_error", attempt=attempt, batch_id=batch_id,
                          error=_safe_error(exc), latency_s=time.monotonic() - started,
                          request_timeout_s=self.request_timeout_s)
                if self._preempt_for_critical(events, batch_id):
                    # Schedule the new critical input, not a retry of failed
                    # feedback. The failed request and its unknown usage remain.
                    with self._condition:
                        self._stats["superseded_errors"].append(_safe_error(exc))
                    return True
                raise
            elapsed = time.monotonic() - started
            usage = {key: value for key in _USAGE_KEYS
                     if type(value := decision.usage.get(key)) is int and value >= 0}
            calls = [asdict(call) for call in decision.calls]
            with self._condition:
                self._stats["completed_responses"] += 1
                for key, value in usage.items():
                    self._stats["usage"][key] = self._stats["usage"].get(key, 0) + value
            self._log("model_response", attempt=attempt, batch_id=batch_id, usage=usage,
                      latency_s=elapsed, calls=calls)
            if self._stopping():
                self._log("discarded_response", attempt=attempt, reason="worker_stopping")
                return False
            if self._preempt_for_critical(events, batch_id):
                self._log("discarded_response", attempt=attempt, reason="new_critical_event")
                return True
            if elapsed > self.request_timeout_s:
                raise ModelError("Late memory response", "MODEL_TIMEOUT")
            if not calls or len(calls) > 16:
                raise ModelProtocolError("Expected a bounded tool response")
            results = []
            committed = False
            for call in calls:
                if len(calls) != 1:
                    result = {"status": "failed", "error_code": "ONE_TOOL_PER_TURN",
                              "details": {"field": "tool_calls", "actual": len(calls), "maximum": 1}}
                else:
                    try:
                        result, selected = self._dispatch(call, snapshot, event_seqs, batch_id, reads)
                        if selected is not None:
                            image_url, image_meta = self._image(selected)
                        committed = call["name"] in {"commit_memory", "patch_memory"} and result.get("status") == "succeeded"
                    except MemoryToolError as exc:
                        result = {"status": "failed", "error_code": exc.code, "details": exc.details}
                    except (StoreError, ModelError, ValueError, TypeError, KeyError) as exc:
                        result = {"status": "failed", "error_code": _safe_error(exc, "INVALID_ARGUMENTS")["code"]}
                result = {"call_id": call["call_id"], "tool": call["name"], **result}
                results.append(result)
                if result.get("status") != "succeeded":
                    with self._condition:
                        self._stats["tool_errors"] += 1
            self._log("tool_results", attempt=attempt, batch_id=batch_id, results=results)
            if any(result.get("error_code") == "VERSION_CONFLICT" for result in results):
                raise ModelError("Batch base revision changed", "MEMORY_VERSION_CONFLICT")
            if committed:
                with self._condition:
                    self._stats["processed_event_seq"] = result["processed_event_seq"]
                    self._stats["committed_batches"] += 1
                    self._condition.notify_all()
                return True
            # Provider-required private fields remain only in bounded RAM.
            history.append({"observation": {"role": "memory_agent", "base_revision": base,
                                            "batch_event_seqs": event_seqs, "current_image": image_meta},
                            "calls": calls, "results": results,
                            "reasoning_content": decision.reasoning_content,
                            "assistant_content": decision.assistant_content})
            history = history[-self.history_turns:] if self.history_turns else []
        return False

    def _preempt_for_critical(self, events, batch_id):
        if any(event["critical"] for event in events):
            return False
        urgent = self.store.snapshot()["pending_critical_events"]
        if not urgent:
            return False
        with self._condition:
            self._stats["preempted_batches"] += 1
        self._log("batch_preempted", batch_id=batch_id,
                  pending_critical_seqs=[event["seq"] for event in urgent])
        return True

    def _dispatch(self, call, snapshot, event_seqs, batch_id, reads):
        name, args = call["name"], call["arguments"]
        validate_tool(name, args)
        base = snapshot["revision"]
        allowed = set(event_seqs) | set(snapshot.get("acknowledged_event_seqs", []))

        def evidence(event_id):
            event = self.store.get_event(event_id)
            if event["seq"] > snapshot["processed_event_seq"] and event["seq"] not in allowed:
                raise ModelError("Event is outside the pinned batch and acknowledged history", "EVENT_OUTSIDE_BATCH")
            return event

        def require_read(path):
            if path not in reads and any(item["path"] == path for item in snapshot["files"]):
                raise MemoryToolError("MEMORY_READ_REQUIRED", {
                    "field": "path", "issue": "read_existing_document_before_edit", "tool": "memory_read"})

        if name == "memory_search":
            return {"status": "succeeded", "matches": self.store.search(
                args["query"], revision=base, limit=args.get("limit", 5))}, None
        if name in {"memory_read", "inspect_event"}:
            if name == "memory_read":
                document = self.store.read(args["path"], revision=base)
                reads.add(args["path"])
                return {"status": "succeeded", "document": document_context(document)}, None
            event = evidence(args["event_id"])
            return {"status": "succeeded", "event": event}, event
        if name == "patch_memory":
            require_read(args["path"])
            document = self.store.read(args["path"], revision=base)
            refs = list(dict.fromkeys(document["evidence_ids"] + args["evidence_ids"]))
            if len(refs) > 16:
                raise MemoryToolError("INVALID_ARGUMENTS", {"field": "evidence_ids",
                    "issue": "patch_preserves_existing_evidence", "actual": len(refs), "max": 16,
                    "recovery": "use_commit_memory_to_explicitly_reconcile_document_and_evidence"})
            changes = [{"path": args["path"], "body": apply_patch_edits(document["body"], args["edits"]),
                        "evidence_ids": refs, "status": args["status"]}]
        else:
            changes = args["changes"]
        for change in changes:
            require_read(change["path"])
            for evidence_id in change["evidence_ids"]:
                evidence(evidence_id)
        receipt_id = hashlib.sha256(call["call_id"].encode("utf-8")).hexdigest()
        result = self.store.commit(base_revision=base, processed_event_seq=max(event_seqs),
                                   changes=changes, event_seqs=event_seqs,
                                   call_id=f"{self._id}:{batch_id}:{receipt_id}")
        return result, None
