"""Bounded text memory and public evidence, with transactional version history.

The agent chooses queries and text. This module supplies literal search, scoped
I/O and hard limits; it neither summarizes episodes nor reads evidence from files.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, name: str, maximum: int, *, empty: bool = False) -> None:
    if (not isinstance(value, str) or len(value) > maximum
            or (not empty and not value.strip())):
        raise ValueError(f"{name} must be {'a' if empty else 'a nonempty'} string of at most {maximum} characters")


def _identifier(value: Any, name: str) -> None:
    _text(value, name, 200)
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{name} cannot contain control characters")


def _positive_integer(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _failure(code: str, message: str) -> dict:
    return {"status": "failed", "error_code": code, "message": message}


class MemoryStoreError(RuntimeError):
    """Sanitized storage failure; messages never include SQL or database paths."""

    def __init__(self, message: str, code: str = "MEMORY_UNAVAILABLE"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class MemoryScope:
    robot_id: str
    controller_id: str
    profile_sha256: str

    def __post_init__(self):
        _identifier(self.robot_id, "robot_id")
        _identifier(self.controller_id, "controller_id")
        if (not isinstance(self.profile_sha256, str) or len(self.profile_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.profile_sha256)):
            raise ValueError("profile_sha256 must be a lowercase SHA256 digest")

    @classmethod
    def from_profile(cls, profile) -> MemoryScope:
        return cls(profile.robot_id, profile.controller_id, _digest(profile.to_dict()))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    max_text_chars: int = 6000
    max_note_chars: int = 4000
    max_query_chars: int = 1000
    max_reason_chars: int = 1000
    max_evidence_refs: int = 8
    max_results: int = 10
    max_excerpt_chars: int = 400
    max_public_evidence_bytes: int = 32768
    max_public_evidence_records: int = 2000
    max_records: int = 200
    max_versions: int = 1000
    max_result_chars: int = 24000

    def __post_init__(self):
        for key, value in asdict(self).items():
            _positive_integer(value, key)

    def to_dict(self) -> dict:
        return asdict(self)


_SCHEMA_VERSION = 1
_SCHEMA = """
BEGIN IMMEDIATE;
CREATE TABLE schema_meta (version INTEGER NOT NULL);
INSERT INTO schema_meta(version) VALUES (1);
CREATE TABLE evidence (
    scope_id TEXT NOT NULL, evidence_ref TEXT NOT NULL,
    episode_id TEXT NOT NULL, payload_json TEXT NOT NULL, sha256 TEXT NOT NULL,
    PRIMARY KEY(scope_id, evidence_ref)
);
CREATE TABLE memories (
    scope_id TEXT NOT NULL, memory_id TEXT NOT NULL, current_version INTEGER NOT NULL,
    PRIMARY KEY(scope_id, memory_id)
);
CREATE TABLE versions (
    scope_id TEXT NOT NULL, memory_id TEXT NOT NULL, version INTEGER NOT NULL,
    text TEXT NOT NULL, reason TEXT NOT NULL, evidence_refs_json TEXT NOT NULL,
    episode_id TEXT NOT NULL, call_id TEXT NOT NULL, created_utc TEXT NOT NULL,
    PRIMARY KEY(scope_id, memory_id, version)
);
CREATE TABLE receipts (
    scope_id TEXT NOT NULL, episode_id TEXT NOT NULL, call_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL, result_json TEXT NOT NULL,
    PRIMARY KEY(scope_id, episode_id, call_id)
);
COMMIT;
"""

_PRIVATE_KEYS = {
    "reasoning_content", "reasoning", "raw_response", "raw_text", "metrics",
    "evaluation", "evaluator", "trajectory", "camera_to_world", "base_xyz_m",
    "target_xy_m", "_target_xy", "target_position", "oracle_success", "agent_context",
}


def _check_public(value: Any) -> None:
    if isinstance(value, dict):
        if any(key in _PRIVATE_KEYS for key in value):
            raise ValueError("Evidence contains private, evaluator or memory-derived fields")
        for item in value.values():
            _check_public(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_public(item)


class MemoryStore:
    """One robot/profile scope and one episode's temporary working note.

    Persistent writes use SQLite transactions, durable call receipts and CAS
    revisions. Working notes and their receipts are RAM-only. Read-only mode
    permits a RAM note and fresh public evidence, while never modifying the DB.
    Off mode never opens, creates or inspects the supplied database path.
    """

    def __init__(self, db_path: str | Path, *, scope: MemoryScope, mode: str = "off",
                 episode_id: str, limits: MemoryLimits | None = None):
        if not isinstance(scope, MemoryScope):
            raise ValueError("An explicit MemoryScope is required")
        if mode not in {"off", "read_only", "read_write"}:
            raise ValueError("Memory mode must be off, read_only or read_write")
        _identifier(episode_id, "episode_id")
        if limits is not None and not isinstance(limits, MemoryLimits):
            raise ValueError("limits must be MemoryLimits")
        self.scope, self.mode, self.episode_id = scope, mode, episode_id
        self.limits = limits if limits is not None else MemoryLimits()
        self._path = Path(db_path)
        self._scope_id = _digest(scope.to_dict())
        self._lock = threading.RLock()
        self._closed = False
        self._working_note, self._note_version = "", 0
        self._ram_receipts: dict[str, tuple[str, dict]] = {}
        self._public_evidence: dict[str, dict] = {}
        if mode == "off":
            return
        try:
            if mode == "read_only" and not self._path.is_file():
                raise MemoryStoreError("Read-only memory requires an existing memory database")
            if mode == "read_write":
                self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables and mode == "read_write":
                    connection.executescript(_SCHEMA)
                version = connection.execute("SELECT version FROM schema_meta").fetchall()
                if version != [(_SCHEMA_VERSION,)]:
                    raise MemoryStoreError("Unsupported memory database schema")
                for table in ("evidence", "memories", "versions", "receipts"):
                    connection.execute(f"SELECT * FROM {table} LIMIT 0")
        except MemoryStoreError:
            raise
        except (OSError, sqlite3.Error, ValueError):
            raise MemoryStoreError("Cannot open or initialize memory database") from None

    @contextmanager
    def _connect(self):
        connection = None
        try:
            if self.mode == "read_only":
                connection = sqlite3.connect(self._path.resolve().as_uri() + "?mode=ro", uri=True, timeout=.25)
                connection.execute("PRAGMA query_only=ON")
            else:
                connection = sqlite3.connect(self._path, timeout=.25)
            yield connection
        except sqlite3.Error as exc:
            if getattr(exc, "sqlite_errorcode", None) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                raise MemoryStoreError("Memory database is busy; no write was committed", "MEMORY_LOCKED") from None
            raise MemoryStoreError("Memory database operation failed") from None
        finally:
            if connection is not None:
                connection.close()

    def _ready(self, *, write: bool = False) -> dict | None:
        if self._closed:
            return _failure("MEMORY_CLOSED", "Memory store is closed")
        if self.mode == "off":
            return _failure("MEMORY_DISABLED", "Memory is disabled")
        if write and self.mode != "read_write":
            return _failure("MEMORY_READ_ONLY", "Persistent memory is read-only")
        return None

    def _bounded(self, result: dict) -> dict:
        if len(_canonical(result)) > self.limits.max_result_chars:
            return _failure("MEMORY_RESULT_LIMIT", "Result exceeds the configured text budget; request a narrower read")
        return result

    def _counts(self, connection) -> tuple[int, int]:
        records = connection.execute("SELECT COUNT(*) FROM memories WHERE scope_id=?", (self._scope_id,)).fetchone()[0]
        versions = connection.execute("SELECT COUNT(*) FROM versions WHERE scope_id=?", (self._scope_id,)).fetchone()[0]
        return records, versions

    def info(self) -> dict:
        with self._lock:
            if self._closed:
                raise MemoryStoreError("Memory store is closed", "MEMORY_CLOSED")
            records, versions = 0, 0
            if self.mode != "off":
                with self._connect() as connection:
                    connection.execute("BEGIN")
                    records, versions = self._counts(connection)
            return {"mode": self.mode, "scope": self.scope.to_dict(), "episode_id": self.episode_id,
                    "record_count": records, "version_count": versions,
                    "working_note": self._working_note, "working_note_version": self._note_version,
                    "limits": self.limits.to_dict()}

    def register_public_evidence(self, evidence_ref: str, public_record: dict) -> dict:
        """Trusted coordinator entry point; never exposed as an agent tool.

        The caller must build a whitelist of public sensor/action feedback. Known
        private fields are rejected as defense against accidentally passing a raw
        history/event object. No paths or original run files are dereferenced.
        """
        try:
            _identifier(evidence_ref, "evidence_ref")
            if not isinstance(public_record, dict):
                raise ValueError("public_record must be an object")
            _check_public(public_record)
            payload = _canonical(public_record)
            if len(payload.encode("utf-8")) > self.limits.max_public_evidence_bytes:
                return _failure("MEMORY_LIMIT", "Public evidence exceeds its byte budget")
        except (ValueError, TypeError, RecursionError, OverflowError):
            return _failure("INVALID_ARGUMENTS", "Invalid or non-public evidence")
        with self._lock:
            unavailable = self._ready()
            if unavailable:
                return unavailable
            item = {"evidence_ref": evidence_ref, "sha256": _digest(public_record), "public_record": json.loads(payload)}
            try:
                if self.mode == "read_write":
                    with self._connect() as connection:
                        connection.execute("BEGIN IMMEDIATE")
                        old = connection.execute("SELECT sha256 FROM evidence WHERE scope_id=? AND evidence_ref=?",
                                                 (self._scope_id, evidence_ref)).fetchone()
                        if old is not None:
                            if old[0] != item["sha256"]:
                                return _failure("EVIDENCE_CONFLICT", "Public evidence references are immutable")
                        else:
                            count = connection.execute("SELECT COUNT(*) FROM evidence WHERE scope_id=?", (self._scope_id,)).fetchone()[0]
                            if count >= self.limits.max_public_evidence_records:
                                return _failure("MEMORY_LIMIT", "Public evidence record limit reached")
                            connection.execute("INSERT INTO evidence VALUES (?,?,?,?,?)",
                                               (self._scope_id, evidence_ref, self.episode_id, payload, item["sha256"]))
                        connection.commit()
                else:
                    old = self._public_evidence.get(evidence_ref)
                    if old is not None and old["sha256"] != item["sha256"]:
                        return _failure("EVIDENCE_CONFLICT", "Public evidence references are immutable")
                    with self._connect() as connection:
                        persisted = connection.execute("SELECT sha256 FROM evidence WHERE scope_id=? AND evidence_ref=?",
                                                       (self._scope_id, evidence_ref)).fetchone()
                    if persisted is not None and persisted[0] != item["sha256"]:
                        return _failure("EVIDENCE_CONFLICT", "Public evidence references are immutable")
                    if old is None and len(self._public_evidence) >= self.limits.max_public_evidence_records:
                        return _failure("MEMORY_LIMIT", "Public evidence record limit reached")
                    self._public_evidence[evidence_ref] = item
            except MemoryStoreError as exc:
                return _failure(exc.code, str(exc))
            return {"status": "succeeded", "evidence_ref": evidence_ref, "sha256": item["sha256"]}

    def search(self, query: str, limit: int = 5) -> dict:
        """Case-insensitive literal substring search of current text, with no ranking policy."""
        try:
            _text(query, "query", self.limits.max_query_chars, empty=True)
            _positive_integer(limit, "limit")
            if limit > self.limits.max_results:
                raise ValueError("limit exceeds max_results")
        except ValueError as exc:
            return _failure("INVALID_ARGUMENTS", str(exc))
        with self._lock:
            unavailable = self._ready()
            if unavailable:
                return unavailable
            try:
                with self._connect() as connection:
                    rows = connection.execute(
                        "SELECT v.memory_id,v.version,v.text,v.evidence_refs_json FROM versions v "
                        "JOIN memories m ON m.scope_id=v.scope_id AND m.memory_id=v.memory_id "
                        "AND m.current_version=v.version WHERE v.scope_id=? ORDER BY v.memory_id", (self._scope_id,)).fetchall()
                matches = [{"memory_id": row[0], "version": row[1], "excerpt": row[2][:self.limits.max_excerpt_chars],
                            "evidence_refs": json.loads(row[3])}
                           for row in rows if query.casefold() in row[2].casefold()][:limit]
                return self._bounded({"status": "succeeded", "matches": matches, "returned": len(matches)})
            except (MemoryStoreError, ValueError, TypeError) as exc:
                return _failure(getattr(exc, "code", "MEMORY_UNAVAILABLE"), "Memory search failed")

    def _version(self, row) -> dict:
        return {"memory_id": row[0], "version": row[1], "text": row[2], "reason": row[3],
                "evidence_refs": json.loads(row[4]), "episode_id": row[5], "call_id": row[6], "created_utc": row[7]}

    def _evidence(self, connection, refs: list[str]) -> list[dict]:
        result = []
        for ref in refs:
            row = connection.execute("SELECT payload_json,sha256 FROM evidence WHERE scope_id=? AND evidence_ref=?",
                                     (self._scope_id, ref)).fetchone()
            if row is None:
                raise MemoryStoreError("Referenced public evidence is missing")
            payload = json.loads(row[0])
            if _digest(payload) != row[1]:
                raise MemoryStoreError("Public evidence integrity check failed")
            result.append({"evidence_ref": ref, "sha256": row[1], "public_record": payload})
        return result

    def read(self, memory_id: str, version: int | None = None) -> dict:
        try:
            _identifier(memory_id, "memory_id")
            if version is not None:
                _positive_integer(version, "version")
        except ValueError as exc:
            return _failure("INVALID_ARGUMENTS", str(exc))
        with self._lock:
            unavailable = self._ready()
            if unavailable:
                return unavailable
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN")
                    current = connection.execute("SELECT current_version FROM memories WHERE scope_id=? AND memory_id=?",
                                                 (self._scope_id, memory_id)).fetchone()
                    if current is None:
                        return _failure("MEMORY_NOT_FOUND", "Memory does not exist in this scope")
                    row = connection.execute(
                        "SELECT memory_id,version,text,reason,evidence_refs_json,episode_id,call_id,created_utc "
                        "FROM versions WHERE scope_id=? AND memory_id=? AND version=?",
                        (self._scope_id, memory_id, version if version is not None else current[0])).fetchone()
                    if row is None:
                        return _failure("MEMORY_NOT_FOUND", "Memory version does not exist in this scope")
                    item = self._version(row)
                    evidence = self._evidence(connection, item["evidence_refs"])
                    versions = [row[0] for row in connection.execute(
                        "SELECT version FROM versions WHERE scope_id=? AND memory_id=? ORDER BY version",
                        (self._scope_id, memory_id))]
                return self._bounded({"status": "succeeded", "memory": item, "current_version": current[0],
                                      "available_versions": versions, "evidence": evidence})
            except (MemoryStoreError, ValueError, TypeError) as exc:
                return _failure(getattr(exc, "code", "MEMORY_UNAVAILABLE"), "Memory read failed")

    def _refs(self, evidence_refs: Any) -> None:
        if (not isinstance(evidence_refs, list) or not 1 <= len(evidence_refs) <= self.limits.max_evidence_refs):
            raise ValueError("evidence_refs must be a nonempty bounded list")
        for ref in evidence_refs:
            _identifier(ref, "evidence_ref")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("evidence_refs must be distinct")

    def _write(self, operation: str, arguments: dict, call_id: str) -> dict:
        with self._lock:
            unavailable = self._ready(write=True)
            if unavailable:
                return unavailable
            fingerprint = _digest([operation, arguments])
            previous = self._ram_receipts.get(call_id)
            if previous is not None:
                return self._replay(previous, fingerprint)
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    old = connection.execute("SELECT fingerprint,result_json FROM receipts WHERE scope_id=? AND episode_id=? AND call_id=?",
                                             (self._scope_id, self.episode_id, call_id)).fetchone()
                    if old is not None:
                        return self._replay((old[0], json.loads(old[1])), fingerprint)
                    # References must already exist in this exact profile scope.
                    if any(connection.execute("SELECT 1 FROM evidence WHERE scope_id=? AND evidence_ref=?",
                                              (self._scope_id, ref)).fetchone() is None
                           for ref in arguments["evidence_refs"]):
                        result = _failure("UNKNOWN_EVIDENCE", "Every reference must name registered public evidence in this scope")
                    else:
                        result = self._apply_write(connection, operation, arguments, call_id)
                    connection.execute("INSERT INTO receipts VALUES (?,?,?,?,?)",
                                       (self._scope_id, self.episode_id, call_id, fingerprint, _canonical(result)))
                    connection.commit()
                return result
            except (MemoryStoreError, ValueError, TypeError) as exc:
                return _failure(getattr(exc, "code", "MEMORY_UNAVAILABLE"), "Memory write was not committed")

    @staticmethod
    def _replay(previous: tuple[str, dict], fingerprint: str) -> dict:
        if previous[0] != fingerprint:
            return _failure("CALL_ID_REUSED", "call_id was already used with different arguments")
        return {**_copy(previous[1]), "deduplicated": True}

    def _apply_write(self, connection, operation: str, arguments: dict, call_id: str) -> dict:
        records, versions = self._counts(connection)
        if versions >= self.limits.max_versions:
            return _failure("MEMORY_LIMIT", "Persistent version limit reached")
        if operation == "save":
            if records >= self.limits.max_records:
                return _failure("MEMORY_LIMIT", "Persistent memory record limit reached")
            memory_id, version, reason = "memory-" + uuid.uuid4().hex, 1, "initial"
        else:
            memory_id = arguments["memory_id"]
            current = connection.execute("SELECT current_version FROM memories WHERE scope_id=? AND memory_id=?",
                                         (self._scope_id, memory_id)).fetchone()
            if current is None:
                return _failure("MEMORY_NOT_FOUND", "Memory does not exist in this scope")
            if current[0] != arguments["expected_version"]:
                return {**_failure("VERSION_CONFLICT", "Memory changed; read the current version before revising"),
                        "current_version": current[0]}
            version, reason = current[0] + 1, arguments["reason"]
        created = _now()
        item = {"memory_id": memory_id, "version": version, "text": arguments["text"], "reason": reason,
                "evidence_refs": arguments["evidence_refs"], "episode_id": self.episode_id,
                "call_id": call_id, "created_utc": created}
        read_result = {"status": "succeeded", "memory": item, "current_version": version,
                       "available_versions": list(range(1, version + 1)),
                       "evidence": self._evidence(connection, arguments["evidence_refs"])}
        if len(_canonical(read_result)) > self.limits.max_result_chars:
            return _failure("MEMORY_RESULT_LIMIT", "Memory and evidence would exceed the read budget; reduce text or evidence references")
        if operation == "save":
            connection.execute("INSERT INTO memories VALUES (?,?,?)", (self._scope_id, memory_id, version))
        else:
            connection.execute("UPDATE memories SET current_version=? WHERE scope_id=? AND memory_id=?",
                               (version, self._scope_id, memory_id))
        connection.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?)",
                           (self._scope_id, memory_id, version, arguments["text"], reason,
                            _canonical(arguments["evidence_refs"]), self.episode_id, call_id, created))
        return {"status": "succeeded", "memory_id": memory_id, "version": version}

    def save(self, text: str, evidence_refs: list[str], *, call_id: str) -> dict:
        try:
            _text(text, "text", self.limits.max_text_chars)
            self._refs(evidence_refs)
            _identifier(call_id, "call_id")
        except ValueError as exc:
            return _failure("INVALID_ARGUMENTS", str(exc))
        return self._write("save", {"text": text, "evidence_refs": list(evidence_refs)}, call_id)

    def revise(self, memory_id: str, expected_version: int, text: str, reason: str,
               evidence_refs: list[str], *, call_id: str) -> dict:
        try:
            _identifier(memory_id, "memory_id")
            _positive_integer(expected_version, "expected_version")
            _text(text, "text", self.limits.max_text_chars)
            _text(reason, "reason", self.limits.max_reason_chars)
            self._refs(evidence_refs)
            _identifier(call_id, "call_id")
        except ValueError as exc:
            return _failure("INVALID_ARGUMENTS", str(exc))
        return self._write("revise", {"memory_id": memory_id, "expected_version": expected_version,
                                     "text": text, "reason": reason, "evidence_refs": list(evidence_refs)}, call_id)

    def update_working_note(self, text: str, *, call_id: str) -> dict:
        try:
            _text(text, "text", self.limits.max_note_chars, empty=True)
            _identifier(call_id, "call_id")
        except ValueError as exc:
            return _failure("INVALID_ARGUMENTS", str(exc))
        with self._lock:
            unavailable = self._ready()
            if unavailable:
                return unavailable
            fingerprint = _digest(["working_note", {"text": text}])
            previous = self._ram_receipts.get(call_id)
            if previous is not None:
                return self._replay(previous, fingerprint)
            try:
                with self._connect() as connection:
                    persisted = connection.execute("SELECT fingerprint,result_json FROM receipts WHERE scope_id=? AND episode_id=? AND call_id=?",
                                                   (self._scope_id, self.episode_id, call_id)).fetchone()
                if persisted is not None:
                    return self._replay((persisted[0], json.loads(persisted[1])), fingerprint)
            except MemoryStoreError as exc:
                return _failure(exc.code, str(exc))
            self._working_note, self._note_version = text, self._note_version + 1
            result = {"status": "succeeded", "working_note": text, "version": self._note_version}
            self._ram_receipts[call_id] = (fingerprint, _copy(result))
            return result

    def snapshot(self) -> dict:
        """Hash logical persistent memory, excluding reads, receipts and temporary state."""
        with self._lock:
            if self._closed:
                raise MemoryStoreError("Memory store is closed", "MEMORY_CLOSED")
            records, evidence = [], []
            if self.mode != "off":
                with self._connect() as connection:
                    # Explicit read transaction gives one consistent concurrent snapshot.
                    connection.execute("BEGIN")
                    rows = connection.execute(
                        "SELECT memory_id,version,text,reason,evidence_refs_json,episode_id,call_id,created_utc "
                        "FROM versions WHERE scope_id=? ORDER BY memory_id,version", (self._scope_id,)).fetchall()
                    referenced, current = set(), None
                    for row in rows:
                        item = self._version(row)
                        if current is None or current["memory_id"] != item["memory_id"]:
                            current = {"memory_id": item["memory_id"], "versions": []}
                            records.append(current)
                        current["versions"].append(item)
                        referenced.update(item["evidence_refs"])
                    evidence = self._evidence(connection, sorted(referenced))
            content = {"schema_version": _SCHEMA_VERSION, "scope": self.scope.to_dict(),
                       "records": records, "evidence": evidence}
            return {"sha256": _digest(content), "content": content}

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._public_evidence.clear()
            self._ram_receipts.clear()
            self._working_note = ""
