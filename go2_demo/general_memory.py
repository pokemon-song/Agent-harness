"""Episode-local public event memory with a caller-owned, on-demand reader.

This module knows no task, target vocabulary, detector, planner or API client.
The caller must explicitly project public data before appending it, and declare
the permitted top-level data fields for each event kind. Nested values of those
fields must already be public; this is not an automatic privacy classifier.

Limits reject whole inputs instead of truncating facts. Reader answers remain
labelled interpretations and are never written back to the event store. API
budgets, timeouts, usage accounting and reader errors belong to the callback.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any


class MemoryLimitError(ValueError):
    """A complete event/query exceeds an explicit configured limit."""


class InvalidMemorySelection(ValueError):
    """The invoked reader returned an invalid selection; nothing was cached."""

    reader_called = True


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(value: Any, *, numeric: bool = False) -> None:
    if numeric and type(value) is int:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Identifiers must be nonempty strings of at most 256 characters (or integer target IDs)")


def _json_value(value: Any, depth: int = 0) -> None:
    """Reject Python-only values rather than silently converting their meaning."""
    if depth > 32:
        raise MemoryLimitError("JSON nesting exceeds 32 levels")
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Event numbers must be finite")
        return
    if type(value) is list:
        for item in value:
            _json_value(item, depth + 1)
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _json_value(item, depth + 1)
        return
    raise ValueError("Event data must contain only JSON values")


def _target_ids(value: Any) -> frozenset[str | int]:
    """Index only explicitly named target metadata, never prose or appearance."""
    found: set[str | int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "target_id" and item is not None:
                _identifier(item, numeric=True)
                found.add(item)
            elif key == "target_ids" and item is not None:
                if not isinstance(item, list):
                    raise ValueError("target_ids metadata must be a JSON list")
                for target in item:
                    _identifier(target, numeric=True)
                    found.add(target)
            if isinstance(item, (dict, list)):
                found.update(_target_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_target_ids(item))
    return frozenset(found)


def _scope(values: Collection | None, *, numeric: bool = False) -> tuple | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Collection):
        raise ValueError("A scope must be a collection of identifiers, not a scalar or object")
    if len(values) > 256:
        raise MemoryLimitError("A query scope may contain at most 256 identifiers")
    for value in values:
        _identifier(value, numeric=numeric)
    return tuple(sorted(set(values), key=_json))


class EpisodeMemory:
    """Append immutable public events and retrieve original evidence by ID.

    ``public_fields`` is a required caller whitelist, e.g.
    ``{"observation": {"pose", "targets"}, "tool_result": {"tool", "result"}}``.
    No source fields are discovered, dropped or copied implicitly.
    Event/store byte limits count canonical UTF-8 JSON payloads; the candidate
    limit counts the complete JSON records supplied to the reader, including IDs.

    An exact repeated (kind, step, observation_id, data) append is idempotent.
    Reusing (kind, step, observation_id) with different data is rejected. The
    same observation ID may occur at different steps or under different kinds.

    A selector receives ``(question, candidates)`` where candidates are complete
    ``{event_id, payload}`` records. Its result must have ``event_ids`` and may
    have ``answer``. Callback exceptions propagate; invalid returned selections
    raise InvalidMemorySelection. Neither failure is cached. ``reader_called``
    means callback invocation, not a claim about billable API requests.
    """

    def __init__(self, *, public_fields: Mapping[str, Collection[str]], max_event_bytes: int = 32768,
                 max_store_bytes: int = 2_000_000, max_candidate_bytes: int = 262144,
                 max_events: int = 10000, max_question_chars: int = 2000,
                 max_answer_chars: int = 4000, max_cache_entries: int = 128):
        limits = {"max_event_bytes": max_event_bytes, "max_store_bytes": max_store_bytes,
                  "max_candidate_bytes": max_candidate_bytes, "max_events": max_events,
                  "max_question_chars": max_question_chars, "max_answer_chars": max_answer_chars,
                  "max_cache_entries": max_cache_entries}
        if any(type(value) is not int or value <= 0 for value in limits.values()):
            raise ValueError("Memory limits must be positive integers")
        if not isinstance(public_fields, Mapping) or not public_fields:
            raise ValueError("Provide an explicit, nonempty public_fields whitelist")
        self._public_fields: dict[str, frozenset[str]] = {}
        for kind, fields in public_fields.items():
            _identifier(kind)
            normalized = _scope(fields)
            if normalized is None:
                raise ValueError("Each event kind requires an explicit data-field whitelist")
            self._public_fields[kind] = frozenset(normalized)
        self._limits = limits
        self._events: list[dict] = []
        self._identities: dict[tuple, dict] = {}
        self._metadata: dict[str, tuple[frozenset, str]] = {}
        self._bytes = 0
        self._cache: OrderedDict[tuple, dict | None] = OrderedDict()

    def append(self, kind: str, step: int, observation_id: str | None, data: dict) -> dict:
        """Append an entire caller-approved JSON event, or reject it unchanged."""
        _identifier(kind)
        if kind not in self._public_fields:
            raise ValueError("Event kind is outside the caller's public whitelist")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if observation_id is not None:
            _identifier(observation_id)
        if type(data) is not dict:
            raise ValueError("data must be a JSON object")
        if set(data) - self._public_fields[kind]:
            raise ValueError("Event data contains fields outside the caller's public whitelist")
        payload = {"kind": kind, "step": step, "observation_id": observation_id, "data": data}
        _json_value(payload)
        targets = _target_ids(data)
        encoded = _json(payload)
        size = len(encoded.encode("utf-8"))
        if size > self._limits["max_event_bytes"]:
            raise MemoryLimitError("Complete event exceeds max_event_bytes; no facts were truncated")
        identity = kind, step, observation_id
        previous = self._identities.get(identity)
        if previous is not None:
            if _json(previous["payload"]) != encoded:
                raise ValueError("Conflicting payload for an existing event identity")
            return deepcopy(previous)
        if len(self._events) >= self._limits["max_events"] or self._bytes + size > self._limits["max_store_bytes"]:
            raise MemoryLimitError("Event store capacity exceeded; no existing facts were removed")
        event = {"event_id": f"E{len(self._events)+1:06d}", "payload": deepcopy(payload)}
        digest = hashlib.sha256(_json(event).encode("utf-8")).hexdigest()
        self._events.append(event)
        self._identities[identity] = event
        self._metadata[event["event_id"]] = targets, digest
        self._bytes += size
        return deepcopy(event)

    def _version(self, events: list[dict]) -> str:
        sources = [(event["event_id"], self._metadata[event["event_id"]][1]) for event in events]
        return hashlib.sha256(_json(sources).encode("utf-8")).hexdigest()

    def _remember(self, key: tuple, card: dict | None) -> None:
        self._cache[key] = deepcopy(card)
        self._cache.move_to_end(key)
        while len(self._cache) > self._limits["max_cache_entries"]:
            self._cache.popitem(last=False)

    def query(self, question: str, selector: Callable | None, event_types: Collection[str] | None = None,
              target_ids: Collection[str | int] | None = None) -> dict:
        """Filter metadata first, then let the caller's reader select up to 24 IDs.

        None means no restriction; an empty scope means no matching events.
        Strings are matched exactly, without target aliases or semantic tests.
        Unrelated appends do not invalidate a scoped result. A new matching
        event changes its evidence version, including for a cached empty result.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonempty string")
        if len(question) > self._limits["max_question_chars"]:
            raise MemoryLimitError("Question exceeds max_question_chars")
        kinds, targets = _scope(event_types), _scope(target_ids, numeric=True)
        candidates = [event for event in self._events
                      if (kinds is None or event["payload"]["kind"] in kinds)
                      and (targets is None or self._metadata[event["event_id"]][0].intersection(targets))]
        version = self._version(candidates)
        key = question, kinds, targets, version
        result = {"evidence_version": version, "candidate_count": len(candidates)}
        if key in self._cache:
            self._cache.move_to_end(key)
            return {**result, "card": deepcopy(self._cache[key]), "cache_hit": True, "reader_called": False}
        if not candidates:
            self._remember(key, None)
            return {**result, "card": None, "cache_hit": False, "reader_called": False}
        if len(_json(candidates).encode("utf-8")) > self._limits["max_candidate_bytes"]:
            raise MemoryLimitError("Complete candidates exceed max_candidate_bytes; narrow the metadata scope")
        if not callable(selector):
            raise TypeError("A callable selector is required for uncached, nonempty evidence")
        frozen = deepcopy(candidates)
        answer = selector(question, deepcopy(frozen))
        if not isinstance(answer, dict) or "event_ids" not in answer or set(answer) - {"event_ids", "answer"}:
            raise InvalidMemorySelection("Reader must return event_ids and an optional answer only")
        ids = answer["event_ids"]
        originals = {event["event_id"]: event for event in frozen}
        if (type(ids) is not list or len(ids) > 24 or any(not isinstance(eid, str) or eid not in originals for eid in ids)
                or len(set(ids)) != len(ids)):
            raise InvalidMemorySelection("Reader selected unknown, duplicate, out-of-scope or more than 24 IDs")
        if "answer" in answer and (not isinstance(answer["answer"], str) or len(answer["answer"]) > self._limits["max_answer_chars"]):
            raise InvalidMemorySelection("Reader answer exceeds max_answer_chars or is not text")
        card = {"question": question, "event_ids": list(ids), "evidence": [deepcopy(originals[eid]) for eid in ids],
                "evidence_version": version, "scope": {"event_types": None if kinds is None else list(kinds),
                                                       "target_ids": None if targets is None else list(targets)}}
        if "answer" in answer:
            card.update(answer=answer["answer"], answer_is_model_generated=True)
        self._remember(key, card)
        return {**result, "card": deepcopy(card), "cache_hit": False, "reader_called": True}

    def snapshot(self) -> dict:
        """Export source facts and limits only, never reader interpretations."""
        return {"schema_version": 1, "event_count": len(self._events), "payload_bytes": self._bytes,
                "evidence_version": self._version(self._events), "events": deepcopy(self._events),
                "public_fields": {kind: sorted(fields) for kind, fields in self._public_fields.items()},
                "limits": deepcopy(self._limits)}

    def write(self, outputdir: str | Path) -> dict[str, Path]:
        """Write one snapshot, replacing each output file atomically."""
        directory = Path(outputdir)
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = self.snapshot()
        markdown = ["# Public episode memory", "", f"Events: {snapshot['event_count']}",
                    f"Evidence version: {snapshot['evidence_version']}", ""]
        for event in snapshot["events"]:
            body = json.dumps(event["payload"], ensure_ascii=False, indent=2, allow_nan=False)
            fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", body)), default=0))
            markdown.extend([f"## {event['event_id']}", "", fence + "json", body, fence, ""])
        documents = {"json": ("memory.json", json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)+"\n"),
                     "markdown": ("memory.md", "\n".join(markdown))}
        result = {}
        for kind, (name, text) in documents.items():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".memory-", suffix=".tmp",
                                                 dir=directory, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(text)
                destination = directory/name
                temporary.replace(destination)
                result[kind] = destination
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return result
