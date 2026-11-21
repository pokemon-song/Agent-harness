"""Ephemeral memory written exclusively from a single live episode's public sensors."""
from copy import deepcopy
import math

from .models import ModelError, _json_text, _usage_counters
from .occluded_handoff import _OutboundNotes
from .occluded_recall import EVIDENCE_LINE, USAGE_KEYS, _tool
from .paired_return_memory import _step
from .return_route_tools import ROUTE_LINE, RouteModel, _sources


SELECT = _tool("select_episode_evidence", "按问题选择本轮有关的证据，路点应按计划行走顺序排列。"
               "只选择有依据的部分，均无记录时可提交空列表。", {
                   "evidence_ids": {"type": "array", "maxItems": 36, "items": {"type": "string"}}})


class FreshMemoryReader(RouteModel):
    def build_messages(self, observation, image_data_url, history):
        return [{"role": "system", "content":
                 "你是独立读者，根据规划器的问题从本轮已发生的公开传感记录选取证据 ID。"
                 "去程中可查询已探索的路点和 A/B 目击。返程可选择从 A 开始按折返顺序的连续机身路点，"
                 "以及最后可靠的目标 B 目击。目标的历史可见表面与机身路点不同。"
                 "记录不完整时选取能回答的部分，不推测新路点或未观测目标。"},
                {"role": "user", "content": _json_text(observation)}]


class FreshEpisodeMemory:
    def __init__(self, mode, task_id, reader=None):
        if mode not in ("empty", "ledger", "query") or (mode == "query" and reader is None):
            raise ValueError("Unknown memory mode or missing reader")
        self.mode, self.reader = mode, reader
        self._notes = None if mode == "empty" else _OutboundNotes(task_id)
        self._card = None
        self._frozen = False
        self._ids = {}
        self._previous_view = None

    def observe(self, harness, step):
        if self._notes is None or self._frozen:
            return
        obs = harness.observation()
        view = (obs["image_id"], tuple((p["point_id"], p.get("valid")) for p in obs["tracked_points"]))
        if view != self._previous_view:
            self._notes.observe(harness, step)
            self._previous_view = view

    def freeze(self):
        if not self._frozen:
            self._frozen = True
            self._card = None

    @property
    def card(self):
        return deepcopy(self._card)

    def documents(self):
        if self._notes is None or self._notes.last_pose is None:
            return {}
        # Original snapshots number all rows afresh. Keep evidence IDs stable when
        # a new target sighting appears before previously presented waypoints.
        docs = {}
        for path, text in self._notes.documents().items():
            lines = []
            for line in text.splitlines():
                match = EVIDENCE_LINE.fullmatch(line)
                if match:
                    excerpt = match.group(2)
                    eid = self._ids.setdefault((path, excerpt), f"E{len(self._ids) + 1:03}")
                    line = f"[{eid}] {excerpt}"
                lines.append(line)
            docs[path] = "\n".join(lines) + "\n"
        return docs

    def _ledger_ids(self, rows):
        route = sorted((row for row in rows.values() if row["path"].startswith("route/")),
                       key=_step, reverse=self._frozen)
        if len(route) > 32:
            route = [route[round(i*(len(route)-1)/31)] for i in range(32)]
        selected = [r["id"] for r in route]
        for label in ("A", "B"):
            sightings = [r for r in rows.values() if r["path"] == f"targets/target-{label}.md"]
            if sightings:
                # Prefer a validated surface point, retaining the most recent
                # color candidate when no tracked observation exists.
                best = max(sightings, key=lambda r: ("有效 RGB-D 跟踪点" in r["excerpt"], _step(r)))
                selected.append(best["id"])
        return selected

    def handle(self, question, *, timeout_s, remaining_requests):
        result = {"status": "unavailable", "error": None, "selection": None, "card": self.card,
                  "requests": 0, "known_usage": {}, "unknown_usage_requests": 0}
        if not isinstance(question, str) or not question.strip() or len(question) > 400:
            return {**result, "status": "error", "error": "INVALID_MEMORY_QUESTION"}
        docs = self.documents()
        if not docs:
            return result
        rows = _sources(docs)
        if self.mode == "ledger":
            ids = self._ledger_ids(rows)
        else:
            if remaining_requests < 1 or not math.isfinite(timeout_s) or timeout_s <= 0:
                return {**result, "status": "error", "error": "MODEL_REQUEST_BUDGET" if remaining_requests < 1 else "MODEL_TIMEOUT"}
            self.reader.raw_usage = None
            result["requests"] = 1
            try:
                response = self.reader.decide({"phase": "return_to_B" if self._frozen else "find_A",
                                               "question": question, "documents": docs}, None, [SELECT], [],
                                              timeout_s=timeout_s)
            except ModelError as exc:
                return {**result, "status": "error", "error": exc.code,
                        **self._account()}
            result.update(self._account())
            if len(response.calls) != 1 or response.calls[0].name != "select_episode_evidence":
                return {**result, "status": "error", "error": "INVALID_MEMORY_SELECTION"}
            args = response.calls[0].arguments
            ids = args.get("evidence_ids") if set(args) == {"evidence_ids"} else None
            if (not isinstance(ids, list) or len(ids) > 36 or
                    not all(isinstance(i, str) and i in rows for i in ids) or len(set(ids)) != len(ids)):
                return {**result, "status": "error", "error": "INVALID_MEMORY_SELECTION"}
        points, sightings = [], []
        for eid in ids:
            row = rows[eid]
            if row["path"].startswith("targets/"):
                sightings.append(row)
            else:
                pose = ROUTE_LINE.fullmatch(row["excerpt"])
                if pose:
                    x, y, yaw = map(float, pose.groups())
                    points.append({"id": eid, "x": x, "y": y, "yaw": yaw})
        if not points and not sightings:
            return result
        focus = "B" if self._frozen else "A"
        target = next((r for r in sightings if r["path"] == f"targets/target-{focus}.md"), None)
        self._card = {"waypoints": points, "sightings": sightings, "target_evidence": target}
        return {**result, "status": "ok", "card": self.card, "selection": {"evidence_ids": ids}}

    def _account(self):
        usage = _usage_counters(self.reader.raw_usage)
        return {"known_usage": usage, "unknown_usage_requests": int(not all(k in usage for k in USAGE_KEYS))}
