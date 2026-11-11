"""Memory schemas, actionable safe errors, and exact local editing semantics."""

from copy import deepcopy
import json
import re

import pytest

from go2_demo.md_memory.contracts import (
    MAX_BODY_CHARS, MAX_TOOL_JSON_BYTES, MemoryToolError,
    apply_patch_edits, memory_tools, validate_tool,
)
from go2_demo.md_memory.store import StoreError, _memory_path


def change(**overrides):
    return {"path": "tasks/order-01.md", "body": "Order ID=order-01; code=001234; revision=2.",
            "evidence_ids": ["event-1"], "status": "confirmed", **overrides}


def error(name, arguments, code):
    with pytest.raises(MemoryToolError) as caught:
        validate_tool(name, arguments)
    assert caught.value.code == code
    return caught.value


@pytest.mark.parametrize("path", [
    "MEMORY.md", "tasks/order.md", "objects/ball_1.md", "places/A-2.md",
    "episodes/session.1.md", "knowledge/a/b/c.md", "tasks/A..B.md",
])
def test_allowed_paths_match_storage_and_public_schema(path):
    validate_tool("memory_read", {"path": path})
    assert _memory_path(path) == path
    schema = next(t for t in memory_tools() if t["function"]["name"] == "memory_read")
    path_schema = schema["function"]["parameters"]["properties"]["path"]
    assert re.fullmatch(path_schema["pattern"], path)
    assert path_schema["maxLength"] == 240


@pytest.mark.parametrize("path", [
    "memo.md", "orders/order.md", "task/order.md", "../tasks/a.md", "/tasks/a.md",
    "tasks/../a.md", "tasks/./a.md", "tasks//a.md", "tasks/a.md/", "tasks/.hidden.md",
    "tasks/a\\b.md", "tasks/订单.md", "tasks/a.txt", "tasks/a.MD", "MEMORY.md/a.md",
])
def test_invalid_paths_match_storage_and_explain_allowed_roots_without_echo(path):
    with pytest.raises(StoreError):
        _memory_path(path)
    exc = error("memory_read", {"path": path}, "INVALID_PATH")
    assert exc.details["field"] == "path"
    assert "tasks" in exc.details["allowed"] and "MEMORY.md" in exc.details["allowed"]
    assert path not in str(exc)


def test_schema_advertises_one_tool_directory_limits_and_plain_body():
    tools = memory_tools()
    assert {tool["function"]["name"] for tool in tools} == {
        "memory_search", "memory_read", "inspect_event", "commit_memory", "patch_memory"}
    commit = next(tool["function"] for tool in tools if tool["function"]["name"] == "commit_memory")
    item = commit["parameters"]["properties"]["changes"]["items"]
    assert item["properties"]["body"]["maxLength"] == MAX_BODY_CHARS
    assert item["properties"]["evidence_ids"]["maxItems"] == 16
    assert commit["parameters"]["properties"]["changes"]["maxItems"] == 4
    for directory in ("tasks/", "objects/", "places/", "episodes/", "knowledge/"):
        assert directory in item["properties"]["path"]["description"]
    assert "frontmatter" in item["properties"]["body"]["description"]
    assert all("exactly one tool" in tool["function"]["description"] for tool in tools)


def test_body_over_limit_reports_exact_field_count_and_max_not_rejected_text():
    rejected = "DO_NOT_ECHO_SECRET_" * 400
    exc = error("commit_memory", {"changes": [change(body=rejected)]}, "BODY_TOO_LONG")
    assert exc.details == {"field": "changes[0].body", "issue": "too_long", "actual": len(rejected), "max": 6000}
    assert "DO_NOT_ECHO_SECRET" not in str(exc)
    validate_tool("commit_memory", {"changes": [change(body="a" * 6000)]})


def test_utf8_total_limit_is_separate_from_body_character_limit():
    body = "汉" * 6000
    assert len(body) == MAX_BODY_CHARS
    exc = error("commit_memory", {"changes": [change(body=body)]}, "TOOL_ARGUMENTS_TOO_LONG")
    assert exc.details["field"] == "arguments"
    assert exc.details["actual"] > MAX_TOOL_JSON_BYTES == exc.details["max"]


@pytest.mark.parametrize("name,args,code,field", [
    ("memory_search", {"query": "x", "limit": True}, "INVALID_FIELD_TYPE", "limit"),
    ("memory_search", {"query": "x", "limit": 11}, "INVALID_RANGE", "limit"),
    ("memory_search", {"query": "x" * 1001}, "TEXT_TOO_LONG", "query"),
    ("memory_read", {"path": 1}, "INVALID_FIELD_TYPE", "path"),
    ("inspect_event", {"event_id": " "}, "EMPTY_TEXT", "event_id"),
    ("inspect_event", {"event_id": "event\n1"}, "INVALID_IDENTIFIER", "event_id"),
    ("commit_memory", {"changes": [change(body=" \n")]}, "EMPTY_TEXT", "changes[0].body"),
    ("commit_memory", {"changes": [change(status="maybe")]}, "INVALID_STATUS", "changes[0].status"),
    ("commit_memory", {"changes": [change(evidence_ids=[])]}, "INVALID_ARRAY_LENGTH", "changes[0].evidence_ids"),
    ("commit_memory", {"changes": [change(evidence_ids=["e", "e"])]}, "DUPLICATE_EVIDENCE", "changes[0].evidence_ids"),
    ("commit_memory", {"changes": [change(), change()]}, "DUPLICATE_PATH", "changes[1].path"),
    ("commit_memory", {"changes": [change() for _ in range(5)]}, "INVALID_ARRAY_LENGTH", "changes"),
])
def test_field_errors_are_actionable(name, args, code, field):
    assert error(name, args, code).details["field"] == field


def test_missing_and_unexpected_fields_do_not_echo_unknown_key_or_value():
    secret = "sk-PRIVATE-DO-NOT-LOG"
    missing = error("memory_read", {}, "MISSING_FIELD")
    assert missing.details["required"] == ["path"]
    unexpected = error("memory_read", {"path": "MEMORY.md", secret: secret}, "UNEXPECTED_FIELD")
    assert unexpected.details["actual"] == 1
    assert secret not in str(unexpected)
    unsupported = error(secret, {}, "UNSUPPORTED_TOOL")
    assert secret not in str(unsupported)


def test_error_constructor_drops_arbitrary_exception_metadata():
    secret = "PRIVATE_SECRET_TOKEN"
    exc = MemoryToolError(secret, {"field": secret, "issue": [secret], "actual": secret,
                                  "message": secret, "allowed": [secret, "tasks"]})
    assert secret not in str(exc) and secret not in json.dumps(exc.details)
    assert exc.code == "INVALID_ARGUMENTS"
    assert exc.details["allowed"] == ["tasks"]


def test_worker_precondition_and_provenance_recovery_survive_safe_error_projection():
    read = MemoryToolError("MEMORY_READ_REQUIRED", {
        "field": "path", "issue": "read_existing_document_before_edit", "tool": "memory_read"})
    assert read.code == "MEMORY_READ_REQUIRED"
    assert read.details == {"field": "path", "issue": "read_existing_document_before_edit", "tool": "memory_read"}
    refs = MemoryToolError("INVALID_ARGUMENTS", {"field": "evidence_ids",
        "issue": "patch_preserves_existing_evidence", "actual": 17, "max": 16,
        "recovery": "use_commit_memory_to_explicitly_reconcile_document_and_evidence"})
    assert refs.details["actual"] == 17 and refs.details["max"] == 16
    assert refs.details["recovery"] == "use_commit_memory_to_explicitly_reconcile_document_and_evidence"
    untrusted = MemoryToolError("INVALID_ARGUMENTS", {"tool": "SECRET_TOOL", "recovery": "SECRET_RECOVERY"})
    assert "SECRET" not in str(untrusted)


def test_validation_is_pure_and_empty_commit_remains_allowed():
    args = {"changes": [change()]}
    original = deepcopy(args)
    assert validate_tool("commit_memory", args) is None
    assert args == original
    validate_tool("commit_memory", {"changes": []})


def test_patch_changes_only_selected_literal_and_retains_order_identity():
    body = "# Dispatch\nOrder=order-001\nRevision=1\nCode=001234\nAlias=冷藏区\nRoute=blue\nEvidence: event-1\n"
    result = apply_patch_edits(body, [{"old": "Revision=1", "new": "Revision=2"},
                                    {"old": "Route=blue", "new": "Route=yellow"}])
    assert result == body.replace("Revision=1", "Revision=2").replace("Route=blue", "Route=yellow")
    for retained in ("Order=order-001", "Code=001234", "Alias=冷藏区", "Evidence: event-1"):
        assert retained in result
    assert "Revision=1" in body  # Input is never modified in place.


def test_patch_sequence_is_literal_not_regex_or_fuzzy_matching():
    result = apply_patch_edits("ID=a.b; status=old", [
        {"old": "a.b", "new": "a*b"}, {"old": "a*b", "new": "a+b"}])
    assert result == "ID=a+b; status=old"
    with pytest.raises(MemoryToolError) as caught:
        apply_patch_edits("ID=axb", [{"old": "a.b", "new": "abc"}])
    assert caught.value.code == "PATCH_NOT_FOUND"
    assert caught.value.details == {"field": "edits[0].old", "issue": "not_found", "matches": 0}


@pytest.mark.parametrize("body,old,matches", [("blue blue", "blue", 2), ("aaa", "aa", 2), ("aaaa", "aa", 3)])
def test_patch_rejects_all_ambiguous_matches_including_overlaps(body, old, matches):
    with pytest.raises(MemoryToolError) as caught:
        apply_patch_edits(body, [{"old": old, "new": "X"}])
    assert caught.value.code == "PATCH_AMBIGUOUS"
    assert caught.value.details["matches"] == matches


def test_patch_fails_atomically_if_a_later_edit_cannot_match():
    body = "Order=001; route=blue; code=000123"
    with pytest.raises(MemoryToolError) as caught:
        apply_patch_edits(body, [{"old": "route=blue", "new": "route=yellow"},
                                {"old": "missing", "new": "replacement"}])
    assert caught.value.code == "PATCH_NOT_FOUND"
    assert caught.value.details["field"] == "edits[1].old"
    assert body == "Order=001; route=blue; code=000123"


def test_patch_enforces_final_body_limit_and_nonempty_body():
    with pytest.raises(MemoryToolError) as caught:
        apply_patch_edits("a" * 5999 + "z", [{"old": "z", "new": "zz"}])
    assert caught.value.code == "BODY_TOO_LONG"
    assert caught.value.details["actual"] == 6001
    with pytest.raises(MemoryToolError) as caught:
        apply_patch_edits("unique", [{"old": "unique", "new": ""}])
    assert caught.value.code == "EMPTY_TEXT"


@pytest.mark.parametrize("edits,code,field", [
    ([], "INVALID_ARRAY_LENGTH", "edits"),
    ([{"old": "x", "new": "y"}] * 9, "INVALID_ARRAY_LENGTH", "edits"),
    ([{"old": "", "new": "y"}], "EMPTY_TEXT", "edits[0].old"),
    ([{"old": "x"}], "MISSING_FIELD", "edits[0]"),
    ([{"old": "x", "new": "y", "regex": True}], "UNEXPECTED_FIELD", "edits[0]"),
])
def test_patch_argument_contract(edits, code, field):
    exc = error("patch_memory", {"path": "tasks/order.md", "edits": edits,
                                "evidence_ids": ["e-1"], "status": "confirmed"}, code)
    assert exc.details["field"] == field


def test_valid_patch_schema_and_fresh_schema_instances():
    validate_tool("patch_memory", {"path": "tasks/order.md", "edits": [{"old": "R1", "new": "R2"}],
                                   "evidence_ids": ["event-2"], "status": "confirmed"})
    tools = memory_tools()
    tools[0]["function"]["parameters"]["properties"]["query"]["maxLength"] = 1
    assert memory_tools()[0]["function"]["parameters"]["properties"]["query"]["maxLength"] == 1000
