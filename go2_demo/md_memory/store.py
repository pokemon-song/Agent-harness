"""Versioned Markdown memory with a transactional event and publication index.

Markdown is the semantic record. SQLite contains public input events, immutable
revision pointers and idempotency receipts, never a second mutable copy of prose.
Each operation opens its own connection; readers see one committed revision.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import uuid


STATUSES = frozenset({"confirmed", "predicted", "unknown", "contradicted", "stale"})
_DIRECTORIES = frozenset({"objects", "places", "tasks", "episodes", "knowledge"})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_PRIVATE = frozenset({"reasoning_content", "chain_of_thought", "scratchpad", "api_key",
                      "authorization", "password", "credentials", "ground_truth", "evaluator"})


class StoreError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise StoreError("INVALID_ARGUMENTS", "Expected finite JSON data") from exc


def _sha(data: bytes):
    return hashlib.sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise StoreError("INVALID_ARGUMENTS", f"{name} must be an integer >= {minimum}")


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(c) < 32 for c in value):
        raise StoreError("INVALID_ARGUMENTS", f"Invalid {name}")


def _public(value):
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, str) and any(word in source.lower() for word in ("privileged", "evaluator", "oracle", "ground_truth")):
            raise StoreError("NON_PUBLIC_EVIDENCE", "Privileged sources cannot become public evidence")
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in _PRIVATE:
                raise StoreError("NON_PUBLIC_EVIDENCE", "Private fields cannot be registered as public evidence")
            _public(item)
    elif isinstance(value, list):
        for item in value:
            _public(item)


def _memory_path(value):
    if not isinstance(value, str) or len(value) > 240 or "\\" in value:
        raise StoreError("INVALID_PATH", "Expected a bounded relative Markdown path")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or not path.parts
            or any(part in {".", ".."} or not _NAME.fullmatch(part) for part in path.parts)
            or not value.endswith(".md")
            or not (value == "MEMORY.md" or (len(path.parts) >= 2 and path.parts[0] in _DIRECTORIES))):
        raise StoreError("INVALID_PATH", "Memory path is outside the allowed Markdown directories")
    return value


def _write_file(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directories(directory: Path):
    """Make directory entries durable before publishing their database pointer."""
    directories = [directory, *(path for path in directory.rglob("*") if path.is_dir())]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True) + [directory.parent]:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class MDMemoryStore:
    def __init__(self, root: Path, scope: str = "demo-world"):
        _identifier(scope, "scope")
        self.root = Path(root).absolute()
        self.scope = scope
        if self.root.resolve() != self.root:
            raise StoreError("INVALID_PATH", "Memory root cannot contain symlinks or parent traversal")
        self.root.mkdir(parents=True, exist_ok=True)
        self._safe(self.root)
        self._versions = self.root / "revisions"
        self._db = self.root / "index.sqlite3"
        self._safe(self._versions)
        self._safe(self._db)
        self._versions.mkdir(exist_ok=True)
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), scope TEXT NOT NULL, head INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS commits (revision INTEGER PRIMARY KEY, directory TEXT NOT NULL UNIQUE, manifest_sha256 TEXT NOT NULL, processed_event_seq INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS receipts (call_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, result TEXT NOT NULL)")
            current = db.execute("SELECT scope FROM settings WHERE id=1").fetchone()
            if current is not None:
                if current[0] != scope:
                    raise StoreError("SCOPE_MISMATCH", "This memory root belongs to a different scope")
            else:
                directory = "00000000-" + uuid.uuid4().hex
                manifest = {"revision": 0, "scope": scope, "processed_event_seq": 0,
                            "acknowledged_event_seqs": [], "files": []}
                data = _json(manifest).encode()
                _write_file(self._versions / directory / "manifest.json", data)
                _sync_directories(self._versions / directory)
                db.execute("INSERT INTO commits VALUES (0,?,?,0)", (directory, _sha(data)))
                db.execute("INSERT INTO settings VALUES (1,?,0)", (scope,))
            db.commit()
        # Reopening validates the publication pointer; abandoned directories are ignored.
        self.snapshot()

    def _safe(self, path: Path):
        try:
            relative = path.relative_to(self.root)
        except ValueError as exc:
            raise StoreError("INVALID_PATH", "Path is outside the memory root") from exc
        if any(part in {".", ".."} for part in relative.parts):
            raise StoreError("INVALID_PATH", "Parent traversal is not permitted in memory storage")
        current = self.root
        if current.is_symlink():
            raise StoreError("INVALID_PATH", "Symlinks are not permitted in memory storage")
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise StoreError("INVALID_PATH", "Symlinks are not permitted in memory storage")
        return path

    @contextmanager
    def _connection(self):
        self._safe(self._db)
        db = None
        try:
            db = sqlite3.connect(self._db, timeout=5, isolation_level=None)
            yield db
        except sqlite3.Error as exc:
            raise StoreError("STORE_UNAVAILABLE", "Memory index operation failed") from exc
        finally:
            if db is not None:
                if db.in_transaction:
                    db.rollback()
                db.close()

    def _image(self, image_path):
        if image_path is None:
            return None
        if (not isinstance(image_path, str) or "\\" in image_path
                or str(PurePosixPath(image_path)) != image_path
                or not image_path.startswith("evidence/")
                or any(p in {".", ".."} for p in PurePosixPath(image_path).parts)
                or PurePosixPath(image_path).suffix.lower() not in {".png", ".jpg", ".jpeg"}):
            raise StoreError("INVALID_PATH", "Images must be relative files under evidence/")
        path = self._safe(self.root / image_path)
        try:
            if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
                raise StoreError("INVALID_EVIDENCE", "Evidence image is missing or exceeds its size budget")
            return _sha(path.read_bytes())
        except OSError as exc:
            raise StoreError("INVALID_EVIDENCE", "Evidence image is unreadable") from exc

    def append_event(self, kind: str, payload: dict, *, image_path: str | None = None, critical: bool = False):
        _identifier(kind, "event kind")
        if not isinstance(payload, dict) or type(critical) is not bool:
            raise StoreError("INVALID_ARGUMENTS", "Expected a public payload and boolean critical flag")
        encoded = _json(payload)
        _public(payload)
        if len(encoded.encode()) > 32768:
            raise StoreError("EVENT_LIMIT", "Event payload exceeds its byte budget")
        event = {"id": "event-" + uuid.uuid4().hex, "kind": kind, "payload": json.loads(_json(payload)),
                 "image_path": image_path, "image_sha256": self._image(image_path),
                 "critical": critical, "observed_at": _now(), "scope": self.scope}
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute("INSERT INTO events(id,payload) VALUES (?,?)", (event["id"], _json(event)))
            event["seq"] = cursor.lastrowid
            db.execute("UPDATE events SET payload=? WHERE seq=?", (_json(event), event["seq"]))
            db.commit()
        return event

    def _event(self, db, event_id):
        row = db.execute("SELECT seq,payload FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None:
            raise StoreError("UNKNOWN_EVIDENCE", "Evidence must refer to a registered event")
        try:
            event = json.loads(row[1])
            if event["seq"] != row[0] or event["id"] != event_id or event["scope"] != self.scope:
                raise ValueError("event identity mismatch")
            if self._image(event["image_path"]) != event["image_sha256"]:
                raise StoreError("EVIDENCE_CORRUPT", "Registered evidence image changed")
            return event
        except (ValueError, KeyError, TypeError) as exc:
            raise StoreError("EVIDENCE_CORRUPT", "Registered event is corrupt") from exc

    def get_event(self, event_id):
        _identifier(event_id, "event id")
        with self._connection() as db:
            return self._event(db, event_id)

    def _load(self, db, revision=None):
        if revision is None:
            revision = db.execute("SELECT head FROM settings WHERE id=1").fetchone()[0]
        _integer(revision, "revision")
        row = db.execute("SELECT directory,manifest_sha256,processed_event_seq FROM commits WHERE revision=?", (revision,)).fetchone()
        if row is None:
            raise StoreError("REVISION_NOT_FOUND", "Requested memory revision does not exist")
        directory = self._safe(self._versions / row[0])
        try:
            raw = self._safe(directory / "manifest.json").read_bytes()
            if _sha(raw) != row[1]:
                raise StoreError("MEMORY_CORRUPT", "Revision manifest checksum mismatch")
            manifest = json.loads(raw)
            if (manifest["revision"] != revision or manifest["scope"] != self.scope
                    or manifest["processed_event_seq"] != row[2] or not isinstance(manifest["files"], list)):
                raise StoreError("MEMORY_CORRUPT", "Revision manifest metadata mismatch")
            # Old revisions are immutable: supply the compatible default only in
            # this decoded value, never rewrite their manifest or its checksum.
            acknowledged = manifest.setdefault("acknowledged_event_seqs", [])
            prefix = manifest["processed_event_seq"]
            if (type(prefix) is not int or prefix < 0 or not isinstance(acknowledged, list)
                    or any(type(seq) is not int or seq <= prefix for seq in acknowledged)
                    or acknowledged != sorted(set(acknowledged))
                    or prefix + 1 in acknowledged):
                raise StoreError("MEMORY_CORRUPT", "Invalid sparse event acknowledgements")
            if acknowledged:
                registered = {item[0] for item in db.execute("SELECT seq FROM events WHERE seq>?", (prefix,))}
                if not set(acknowledged).issubset(registered):
                    raise StoreError("MEMORY_CORRUPT", "Acknowledged event does not exist")
            contents, paths = {}, set()
            for entry in manifest["files"]:
                path = _memory_path(entry["path"])
                if path in paths:
                    raise StoreError("MEMORY_CORRUPT", "Duplicate document in manifest")
                paths.add(path)
                data = self._safe(directory / path).read_bytes()
                if _sha(data) != entry["sha256"]:
                    raise StoreError("MEMORY_CORRUPT", "Markdown checksum mismatch")
                contents[path] = data
            return manifest, contents
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StoreError("MEMORY_CORRUPT", "Committed memory revision is unreadable") from exc

    @staticmethod
    def _pending(db, manifest):
        acknowledged = set(manifest["acknowledged_event_seqs"])
        return [json.loads(row[1]) for row in db.execute(
            "SELECT seq,payload FROM events WHERE seq>? ORDER BY seq", (manifest["processed_event_seq"],))
                if row[0] not in acknowledged]

    def snapshot(self, revision: int | None = None):
        with self._connection() as db:
            db.execute("BEGIN")
            manifest, _ = self._load(db, revision)
            latest = db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
            pending = self._pending(db, manifest)
            acknowledged = set(manifest["acknowledged_event_seqs"])
            processed_critical = next((seq for seq, payload in db.execute(
                "SELECT seq,payload FROM events ORDER BY seq DESC")
                if (seq <= manifest["processed_event_seq"] or seq in acknowledged)
                and json.loads(payload)["critical"]), 0)
            return {**manifest, "latest_event_seq": latest, "pending_event_count": len(pending),
                    "processed_critical_event_seq": processed_critical,
                    "pending_critical_events": [{"seq": e["seq"], "id": e["id"], "kind": e["kind"]} for e in pending if e["critical"]]}

    def pending_events(self, limit=8, *, prioritize_critical=False):
        _integer(limit, "limit", 1)
        if limit > 100 or type(prioritize_critical) is not bool:
            raise StoreError("INVALID_ARGUMENTS", "Event batch limit cannot exceed 100")
        with self._connection() as db:
            db.execute("BEGIN")
            manifest, _ = self._load(db)
            pending = self._pending(db, manifest)
            if prioritize_critical:
                critical = [event for event in pending if event["critical"]]
                pending = critical or pending
            return [self._event(db, event["id"]) for event in pending[:limit]]

    def is_event_processed(self, seq):
        """Whether an event is acknowledged, including outside the FIFO prefix."""
        _integer(seq, "seq", 1)
        with self._connection() as db:
            db.execute("BEGIN")
            manifest, _ = self._load(db)
            return seq <= manifest["processed_event_seq"] or seq in manifest["acknowledged_event_seqs"]

    def read(self, path, revision=None):
        _memory_path(path)
        with self._connection() as db:
            db.execute("BEGIN")
            manifest, contents = self._load(db, revision)
            entry = next((item for item in manifest["files"] if item["path"] == path), None)
            if entry is None:
                raise StoreError("MEMORY_NOT_FOUND", "Document does not exist in the requested revision")
            content = contents[path].decode("utf-8")
            return {**entry, "snapshot_revision": manifest["revision"], "content": content,
                    "body": content.split("\n---\n", 1)[1]}

    def search(self, query, revision=None, limit=5):
        if not isinstance(query, str) or len(query) > 1000:
            raise StoreError("INVALID_ARGUMENTS", "Search query must be bounded text")
        _integer(limit, "limit", 1)
        if limit > 20:
            raise StoreError("INVALID_ARGUMENTS", "Search limit cannot exceed 20")
        with self._connection() as db:
            db.execute("BEGIN")
            manifest, contents = self._load(db, revision)
            result = []
            for entry in manifest["files"]:
                body = contents[entry["path"]].decode().split("\n---\n", 1)[1]
                if query.casefold() in (entry["path"] + "\n" + body).casefold():
                    result.append({**entry, "summary": body[:400]})
                    if len(result) == limit:
                        break
            return result

    def commit(self, base_revision: int, processed_event_seq: int, changes: list[dict], *, call_id: str,
               event_seqs: list[int] | None = None):
        """Publish prose and acknowledge an input window atomically.

        Without event_seqs the watermark retains its legacy prefix semantics.
        An explicit list acknowledges only those pending events; the supplied
        watermark is their maximum, while the returned watermark is the newly
        contiguous prefix. Previously acknowledged events remain usable evidence.
        """
        _integer(base_revision, "base_revision")
        _integer(processed_event_seq, "processed_event_seq")
        _identifier(call_id, "call_id")
        if event_seqs is not None:
            if (not isinstance(event_seqs, list) or not 1 <= len(event_seqs) <= 100
                    or any(type(seq) is not int or seq < 1 for seq in event_seqs)
                    or event_seqs != sorted(set(event_seqs))):
                raise StoreError("INVALID_ARGUMENTS", "Selected events must be 1..100 distinct increasing positive sequences")
            if processed_event_seq != event_seqs[-1]:
                raise StoreError("INVALID_WATERMARK", "Selected window maximum does not match its watermark")
        if not isinstance(changes, list) or len(changes) > 4:
            raise StoreError("INVALID_ARGUMENTS", "A commit permits at most four document changes")
        paths = set()
        for change in changes:
            if not isinstance(change, dict) or set(change) != {"path", "body", "evidence_ids", "status"}:
                raise StoreError("INVALID_ARGUMENTS", "Invalid memory change fields")
            path = _memory_path(change["path"])
            if path in paths:
                raise StoreError("INVALID_ARGUMENTS", "A document can only be changed once per commit")
            paths.add(path)
            if not isinstance(change["body"], str) or not change["body"].strip() or len(change["body"]) > 6000:
                raise StoreError("INVALID_ARGUMENTS", "Markdown body must contain 1..6000 characters")
            if not isinstance(change["status"], str) or change["status"] not in STATUSES:
                raise StoreError("INVALID_ARGUMENTS", "Invalid epistemic status")
            refs = change["evidence_ids"]
            if not isinstance(refs, list) or not 1 <= len(refs) <= 16:
                raise StoreError("INVALID_ARGUMENTS", "Changes need 1..16 registered evidence IDs")
            for ref in refs:
                _identifier(ref, "evidence id")
            if len(set(refs)) != len(refs):
                raise StoreError("INVALID_ARGUMENTS", "Evidence IDs cannot repeat")
        request = {"base_revision": base_revision, "processed_event_seq": processed_event_seq, "changes": changes}
        # Preserve fingerprints of commits recorded by the legacy API.
        if event_seqs is not None:
            request["event_seqs"] = event_seqs
        fingerprint = _sha(_json(request).encode())
        directory = None
        published = False
        try:
            with self._connection() as db:
                db.execute("BEGIN IMMEDIATE")
                receipt = db.execute("SELECT fingerprint,result FROM receipts WHERE call_id=?", (call_id,)).fetchone()
                if receipt:
                    if receipt[0] != fingerprint:
                        raise StoreError("CALL_ID_REUSED", "Commit ID was reused with different contents")
                    return {**json.loads(receipt[1]), "deduplicated": True}
                before, contents = self._load(db)
                if before["revision"] != base_revision:
                    raise StoreError("VERSION_CONFLICT", "Memory head changed; read a fresh snapshot")
                latest = db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
                if not before["processed_event_seq"] <= processed_event_seq <= latest:
                    raise StoreError("INVALID_WATERMARK", "Processed event watermark must be monotone and registered")
                previous_prefix = before["processed_event_seq"]
                acknowledged = set(before["acknowledged_event_seqs"])
                selected = set(event_seqs or [])
                if event_seqs is None:
                    prefix = processed_event_seq
                    acknowledged = {seq for seq in acknowledged if seq > prefix}
                else:
                    for seq in event_seqs:
                        if seq <= previous_prefix or seq in acknowledged:
                            raise StoreError("EVENT_ALREADY_PROCESSED", "Selected event is no longer pending")
                        event_row = db.execute("SELECT id FROM events WHERE seq=?", (seq,)).fetchone()
                        if event_row is None:
                            raise StoreError("UNKNOWN_EVIDENCE", "Selected event must be registered")
                        self._event(db, event_row[0])
                    prefix = previous_prefix
                    acknowledged.update(selected)
                while prefix + 1 in acknowledged:
                    prefix += 1
                    acknowledged.remove(prefix)
                entries = {item["path"]: item for item in before["files"]}
                revision = base_revision + 1
                for change in changes:
                    events = [self._event(db, ref) for ref in change["evidence_ids"]]
                    maximum = max(event["seq"] for event in events)
                    if any(event["seq"] > prefix and event["seq"] not in acknowledged for event in events):
                        raise StoreError("FUTURE_EVIDENCE", "Evidence is neither acknowledged nor selected by this commit")
                    previous = entries.get(change["path"])
                    if previous and maximum < previous["evidence_max_seq"]:
                        raise StoreError("STALE_EVIDENCE", "Older evidence cannot overwrite a newer document")
                    metadata = {"path": change["path"], "revision": revision, "scope": self.scope,
                                "status": change["status"], "evidence_ids": change["evidence_ids"],
                                "evidence_max_seq": maximum, "observed_at": max(e["observed_at"] for e in events),
                                "evidence": [{"id": e["id"], "seq": e["seq"], "observed_at": e["observed_at"]} for e in events]}
                    # JSON values are valid YAML scalars/flow collections. No YAML parser or executable tags.
                    frontmatter = "\n".join(f"{key}: {_json(value)}" for key, value in metadata.items())
                    data = ("---\n" + frontmatter + "\n---\n" + change["body"]).encode("utf-8")
                    entries[change["path"]] = {**metadata, "sha256": _sha(data)}
                    contents[change["path"]] = data
                manifest = {"revision": revision, "scope": self.scope, "processed_event_seq": prefix,
                            "acknowledged_event_seqs": sorted(acknowledged),
                            "files": [entries[path] for path in sorted(entries)]}
                name = f"{revision:08d}-" + uuid.uuid4().hex
                directory = self._safe(self._versions / name)
                for path, data in contents.items():
                    _write_file(self._safe(directory / path), data)
                encoded = _json(manifest).encode()
                _write_file(directory / "manifest.json", encoded)
                _sync_directories(directory)
                # The directory is complete before a transaction makes its pointer visible.
                result = {"status": "succeeded", "revision": revision, "processed_event_seq": prefix,
                          "acknowledged_event_seqs": sorted(acknowledged)}
                db.execute("INSERT INTO commits VALUES (?,?,?,?)", (revision, name, _sha(encoded), prefix))
                db.execute("UPDATE settings SET head=? WHERE id=1", (revision,))
                db.execute("INSERT INTO receipts VALUES (?,?,?)", (call_id, fingerprint, _json(result)))
                db.commit()
                published = True
                return result
        except OSError as exc:
            raise StoreError("STORE_UNAVAILABLE", "Memory revision could not be written") from exc
        finally:
            if directory is not None and not published and directory.exists() and not directory.is_symlink():
                try:
                    shutil.rmtree(directory)
                except OSError:
                    # An unreachable, unindexed directory is harmless and never becomes head.
                    pass
