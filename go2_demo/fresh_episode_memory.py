"""Ephemeral memory written exclusively from a single live episode's public sensors."""
from copy import deepcopy
import math
import re

from .models import ModelError, _json_text, _usage_counters
from .occluded_handoff import _OutboundNotes
from .occluded_recall import EVIDENCE_LINE, USAGE_KEYS, _tool
from .paired_return_memory import _step
from .return_route_tools import ROUTE_LINE, RouteModel, _sources


SELECT = _tool("select_episode_evidence", "按问题选择本轮有关的证据，路点应按计划行走顺序排列。"
               "只选择有依据的部分，均无记录时可提交空列表。", {
                   "evidence_ids": {"type": "array", "maxItems": 36, "items": {"type": "string"}}})


INTENTS = ("a_sighting", "b_sighting", "route_outbound", "route_reverse", "scene_coverage")


def _legacy_intents(question, frozen):
    """Infer broad fields only for older, question-only tool calls."""
    fields = []
    for label, colour in (("a", "蓝球"), ("b", "绿方块")):
        if re.search(rf"(?<![a-z]){label}(?![a-z])", question, re.I) or colour in question:
            fields.append(f"{label}_sighting")
    if not fields and "目标" in question:
        fields.extend(("a_sighting", "b_sighting"))
    if any(word in question for word in ("路线", "路点", "走过", "返程", "route", "waypoint")):
        fields.append("route_reverse" if frozen or "逆序" in question else "route_outbound")
    if any(word in question for word in ("障碍", "通行", "覆盖", "通道", "开口")):
        fields.append("scene_coverage")
    return fields or (["b_sighting", "route_reverse"] if frozen else ["a_sighting", "route_outbound"])


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
        self._notes = None if mode == "empty" else _OutboundNotes(task_id, sample_distinct_views=True)
        self._card = None
        self._frozen = False
        self._ids = {}
        self._previous_view = None
        self._cache = {}
        self._consecutive_reads = 0
        self._requires_motion = False

    def observe(self, harness, step):
        if self._notes is None or self._frozen:
            return
        obs = harness.observation()
        view = (obs["image_id"], tuple((p["point_id"], p.get("valid")) for p in obs["tracked_points"]))
        if view != self._previous_view:
            self._notes.observe(harness, step)
            self._previous_view = view

    def freeze(self, *, preserve_context=False):
        """Freeze outbound evidence, optionally retaining its last answer and cache.

        Retention changes only the stage-boundary reset. Existing evidence
        versions still govern cache validity when a subsequent query is made.
        """
        if not self._frozen:
            self._frozen = True
            if not preserve_context:
                self._card = None
                self._cache.clear()
                self.after_physical_action()

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

    def query_access(self):
        if self.mode == "empty":
            reason = "no_external_memory"
        elif self._requires_motion or self._consecutive_reads >= len(INTENTS):
            reason = "choose_a_physical_action_before_querying_again"
        else:
            reason = None
        return {"allowed": reason is None, "intents": list(INTENTS) if reason is None else [], "reason": reason}

    def after_physical_action(self):
        self._consecutive_reads = 0
        self._requires_motion = False

    def _eligible(self, intent, rows, route_leg, sighting_window):
        if intent.startswith("route_"):
            leg_path = f"route/leg-{route_leg:02d}.md"
            # Last pose changes with every physics tick. Use recorded route anchors
            # during live search and include the exact final pose at handoff.
            anchors = {p.step for p in self._notes.route} if self._notes and not self._frozen else None
            return [r for r in rows.values() if r["path"].startswith("route/")
                    and (not route_leg or r["path"] == leg_path)
                    and (anchors is None or _step(r) in anchors)]
        if intent.endswith("_sighting"):
            label = intent[0].upper()
            sightings = [r for r in rows.values() if r["path"] == f"targets/target-{label}.md"]
            if sightings and sighting_window != "all":
                pick = min if sighting_window == "first" else max
                step = pick(_step(r) for r in sightings)
                sightings = [r for r in sightings if _step(r) == step]
            return sightings
        # The underlying log contains no obstacle survey or negative visibility map.
        return []

    @staticmethod
    def _ledger_ids(intent, eligible):
        if intent.startswith("route_"):
            ordered = sorted(eligible, key=_step, reverse=intent == "route_reverse")
            if len(ordered) > 32:
                ordered = [ordered[round(i*(len(ordered)-1)/31)] for i in range(32)]
            return [r["id"] for r in ordered]
        best = max(eligible, key=lambda r: ("有效 RGB-D 跟踪点" in r["excerpt"], _step(r)))
        return [best["id"]]

    def _select(self, question, pending, route_leg, sighting_window, rows, result, timeout_s, remaining_requests):
        if remaining_requests < 1:
            result.update(status="error", error="MODEL_REQUEST_BUDGET")
            return None
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            result.update(status="error", error="MODEL_TIMEOUT")
            return None
        docs = {}
        for row in rows.values():
            path = row["path"]
            docs.setdefault(path, []).append(f"[{row['id']}] {row['excerpt']}")
        docs = {path: "# " + path + "\n\n" + "\n".join(lines) + "\n" for path, lines in docs.items()}
        self.reader.raw_usage = None
        result["requests"] = 1
        try:
            response = self.reader.decide({"phase": "return_to_B" if self._frozen else "find_A",
                                           "question": question, "fields_to_select": pending, "route_leg": route_leg,
                                           "sighting_window": sighting_window, "documents": docs},
                                          None, [SELECT], [], timeout_s=timeout_s)
        except ModelError as exc:
            result.update(status="error", error=exc.code)
            return None
        finally:
            result.update(self._account())
        if len(response.calls) == 1 and response.calls[0].name == "select_episode_evidence":
            args = response.calls[0].arguments
            ids = args.get("evidence_ids") if isinstance(args, dict) and set(args) == {"evidence_ids"} else None
            if (isinstance(ids, list) and len(ids) <= 36 and all(isinstance(i, str) and i in rows for i in ids)
                    and len(set(ids)) == len(ids)):
                return ids
        result.update(status="error", error="INVALID_MEMORY_SELECTION")
        return None

    def handle(self, question, *, timeout_s, remaining_requests, intents=None, route_leg=0, sighting_window="all"):
        self._card = None
        result = {"status": "unavailable", "error": None, "selection": None, "card": None, "items": {},
                  "cache_hit": False, "cached_intents": [], "requests": 0, "known_usage": {}, "unknown_usage_requests": 0}
        if not isinstance(question, str) or not question.strip() or len(question) > 400:
            return {**result, "status": "error", "error": "INVALID_MEMORY_QUESTION"}
        if intents is None:
            intents = _legacy_intents(question, self._frozen)
        if (not isinstance(intents, list) or not 1 <= len(intents) <= len(INTENTS)
                or not all(isinstance(i, str) and i in INTENTS for i in intents) or len(intents) != len(set(intents))
                or type(route_leg) is not int or route_leg < 0 or sighting_window not in ("all", "first", "latest")):
            return {**result, "status": "error", "error": "INVALID_MEMORY_SCOPE"}
        if self.mode == "empty":
            result["items"] = {intent: {"status": "memory_disabled", "evidence_ids": []} for intent in intents}
            return result
        if not self.query_access()["allowed"]:
            return {**result, "status": "error", "error": "MEMORY_QUERY_PAUSED"}
        self._consecutive_reads += 1
        rows = _sources(self.documents())
        pending, scopes, eligible_ids = [], {}, {}
        items = result["items"]
        for intent in intents:
            eligible = self._eligible(intent, rows, route_leg, sighting_window)
            ids = tuple(r["id"] for r in eligible)
            scope_arg = route_leg if intent.startswith("route_") else sighting_window if intent.endswith("_sighting") else None
            key = (intent, scope_arg)
            scopes[intent] = key
            eligible_ids[intent] = ids
            cached = self._cache.get(key)
            if cached is not None and cached[0] == ids:
                items[intent] = deepcopy(cached[1])
                result["cached_intents"].append(intent)
            elif ids:
                pending.append(intent)
            else:
                item = {"status": "no_record", "evidence_ids": []}
                items[intent] = item
                self._cache[key] = (ids, item)
        if pending:
            if self.mode == "query":
                relevant = {eid: rows[eid] for intent in pending for eid in eligible_ids[intent]}
                chosen = self._select(question, pending, route_leg, sighting_window, relevant,
                                      result, timeout_s, remaining_requests)
                if chosen is None:
                    for intent in pending:
                        items[intent] = {"status": "reader_error", "evidence_ids": []}
                    return result
            for intent in pending:
                allowed_ids = eligible_ids[intent]
                if self.mode == "ledger":
                    ids = self._ledger_ids(intent, [rows[eid] for eid in allowed_ids])
                else:
                    ids = [eid for eid in chosen if eid in allowed_ids]
                    if intent.startswith("route_"):
                        ids.sort(key=lambda eid: _step(rows[eid]), reverse=intent == "route_reverse")
                item = {"status": "available" if ids else "not_selected", "evidence_ids": ids}
                items[intent] = item
                self._cache[scopes[intent]] = (allowed_ids, item)
        result["cache_hit"] = len(result["cached_intents"]) == len(intents)
        if result["cache_hit"]:
            self._requires_motion = True
        # A mixed request can ask for both directions. Per-field lists always
        # preserve each order; the compact waypoint card follows the stage direction.
        preferred = "route_reverse" if self._frozen else "route_outbound"
        ordered = sorted(intents, key=lambda i: i != preferred)
        ids = list(dict.fromkeys(eid for intent in ordered for eid in items[intent]["evidence_ids"]))
        result["selection"] = {"evidence_ids": ids}
        if not ids:
            return result
        points, sightings = [], []
        for eid in ids:
            row = rows[eid]
            if row["path"].startswith("targets/"):
                sightings.append(row)
            else:
                x, y, yaw = map(float, ROUTE_LINE.fullmatch(row["excerpt"]).groups())
                points.append({"id": eid, "x": x, "y": y, "yaw": yaw})
        focus = "B" if self._frozen else "A"
        target = next((r for r in sightings if r["path"] == f"targets/target-{focus}.md"), None)
        self._card = {"waypoints": points, "sightings": sightings, "target_evidence": target, "items": deepcopy(items)}
        result["status"] = "ok" if all(v["status"] == "available" for v in items.values()) else "partial"
        result["card"] = self.card
        return result

    def _account(self):
        usage = _usage_counters(self.reader.raw_usage)
        return {"known_usage": usage, "unknown_usage_requests": int(not all(k in usage for k in USAGE_KEYS))}
