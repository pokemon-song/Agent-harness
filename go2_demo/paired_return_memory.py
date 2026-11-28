"""One outbound memory slot with shared, source-checked card representation."""
from copy import deepcopy
import math
import re

from .models import ModelError, _usage_counters
from .occluded_recall import USAGE_KEYS
from .return_route_tools import ABORT, B_SIGHTING, ROUTE_LINE, SELECT, _selected, _sources


def _step(row):
    return int(re.match(r"去程步 (\d+)", row["excerpt"]).group(1))


def _ledger_ids(rows):
    route = sorted((row for row in rows.values() if re.fullmatch(r"route/leg-\d+\.md", row["path"])
                    and ROUTE_LINE.fullmatch(row["excerpt"])), key=_step, reverse=True)
    # Outbound recording has already sampled by odometry. Keep both ends if
    # a longer recording needs to fit the same 32-waypoint interface.
    if len(route) > 32:
        route = [route[round(i * (len(route) - 1) / 31)] for i in range(32)]
    sightings = [row for row in rows.values() if row["path"] == "targets/target-B.md"
                 and B_SIGHTING.match(row["excerpt"]) and "有效 RGB-D 跟踪点" in row["excerpt"]]
    target = max(sightings, key=_step, default=None)
    if route and target:
        return {"route_ids": [row["id"] for row in route], "target_id": target["id"]}
    return None


class PairedReturnMemory:
    """Expose a card after a query; model-reader calls are charged once per attempt."""

    def __init__(self, mode, documents, reader=None):
        if mode not in {"empty", "ledger", "query"}:
            raise ValueError("Unknown outbound memory mode")
        if mode == "query" and reader is None:
            raise ValueError("Query mode requires a memory reader")
        self.mode = mode
        self.reader = reader if mode == "query" else None
        self._docs = dict(documents) if mode != "empty" else {}
        self._rows = _sources(self._docs)
        self._card = None

    @property
    def card(self):
        return deepcopy(self._card)

    def handle(self, question, *, timeout_s, remaining_requests):
        result = {"status": "unavailable", "error": None, "selection": None, "card": self.card,
                  "requests": 0, "known_usage": {}, "unknown_usage_requests": 0}
        if not isinstance(question, str) or not question.strip() or len(question) > 400:
            return {**result, "status": "error", "error": "INVALID_MEMORY_QUESTION"}
        if self.mode == "empty":
            return result
        if self.mode == "ledger":
            selected = _ledger_ids(self._rows)
            if selected is None:
                return result
        else:
            if remaining_requests < 1:
                return {**result, "status": "error", "error": "MODEL_REQUEST_BUDGET"}
            if not math.isfinite(timeout_s) or timeout_s <= 0:
                return {**result, "status": "error", "error": "MODEL_TIMEOUT"}
            result["requests"] = 1
            self.reader.raw_usage = self.reader.response_finish_reason = None
            inputs = {"question": question, "documents": self._docs}
            try:
                decision = self.reader.decide({"stage": "read", "input": inputs}, None, [SELECT, ABORT], [],
                                              timeout_s=timeout_s)
            except ModelError as exc:
                result.update(status="error", error=exc.code)
                return result
            finally:
                usage = _usage_counters(self.reader.raw_usage)
                result.update(known_usage=usage, unknown_usage_requests=int(not all(k in usage for k in USAGE_KEYS)))
            if len(decision.calls) != 1:
                return {**result, "status": "error", "error": "INVALID_ROUTE_SELECTION"}
            call = decision.calls[0]
            if call.name == "abort" and set(call.arguments) == {"reason"}:
                return {**result, "selection": {"tool": "abort"}}
            if call.name != SELECT["function"]["name"] or set(call.arguments) != {"route_ids", "target_id"}:
                return {**result, "status": "error", "error": "INVALID_ROUTE_SELECTION"}
            selected = call.arguments
        try:
            points, target = _selected(selected, self._rows)
        except ValueError:
            return {**result, "status": "error", "error": "INVALID_ROUTE_SELECTION"}
        self._card = {"waypoints": points, "target_evidence": deepcopy(target)}
        result.update(status="ok", card=self.card, selection=deepcopy(selected))
        return result
