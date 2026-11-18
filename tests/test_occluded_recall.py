import json

import httpx
import pytest

from go2_demo.occluded_recall import Recall, RecallResponder


DOCS = {"first.md": "# Observations\n- [E001] A was visible in frame 1.",
        "second.md": "[E002a] B was seen once in frame 2; its location needs verification."}
USAGE = {"prompt_tokens": 114, "completion_tokens": 16, "total_tokens": 130}


def responder(handler):
    return RecallResponder("https://example.invalid/", "mock", "", trust_env=False,
                           transport=httpx.MockTransport(handler))


def response(selection, *, usage=USAGE, as_json=False):
    if as_json:
        message = {"content": json.dumps(selection)}
    else:
        message = {"tool_calls": [{"id": "select-1", "type": "function",
                   "function": {"name": "select_evidence", "arguments": json.dumps(selection)}}]}
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": message}], "usage": usage})


def test_direct_only_reads_one_immutable_document_and_persists_it():
    source = DOCS.copy()
    recall = Recall(source, "direct")
    source["first.md"] = "changed"
    assert not recall.read_ready and not recall.needs_request("memory_read", {"path": "first.md"})
    result = recall.handle("memory_read", {"path": "first.md"}, timeout_s=5)
    assert result == {"tool": "memory_read", "status": "succeeded"} and recall.read_ready
    assert recall.context()["current"] == {"path": "first.md", "content": DOCS["first.md"]}
    assert recall.handle("memory_read", {"path": "../stray"}, timeout_s=5)["status"] == "failed"
    assert recall.context()["current"]["path"] == "first.md" and recall.stats["requests"] == 0


def test_selected_quotes_are_exact_and_cached_without_repeating_content_in_tool_feedback():
    calls = []

    def handler(req):
        payload = json.loads(req.content)
        assert [t["function"]["name"] for t in payload["tools"]] == ["select_evidence"]
        assert [m["role"] for m in payload["messages"]] == ["system", "user"]
        calls.append(json.loads(payload["messages"][-1]["content"]))
        return response({"support": "partial", "ids": ["E002a"]})

    recall = Recall(DOCS, "query", responder(handler))
    assert recall.needs_request("memory_query", {"question": "Where was B?"})
    result = recall.handle("memory_query", {"question": "Where was B?"}, timeout_s=5)
    assert result == {"tool": "memory_query", "status": "succeeded", "cache_hit": False}
    assert calls == [{"question": "Where was B?", "documents": DOCS}]
    answer = recall.context()["current"]
    assert answer == {"question": "Where was B?", "support": "partial", "evidence": [
        {"id": "E002a", "path": "second.md", "excerpt": DOCS["second.md"].split("] ", 1)[1]}]}
    answer["evidence"].clear()
    assert not recall.needs_request("memory_query", {"question": "Where  was B? "})
    assert recall.handle("memory_query", {"question": "Where  was B? "}, timeout_s=5)["cache_hit"]
    assert recall.context()["current"]["evidence"] and len(calls) == 1
    assert recall.stats["known_usage"] == USAGE and recall.stats["usage_complete"]


@pytest.mark.parametrize("invalid", [
    {"support": "supported", "ids": ["E999"]}, {"support": "supported", "ids": []},
    {"support": "partial", "ids": ["E001", "E001"]}, {"support": "insufficient", "ids": ["E001"]},
    {"support": "partial", "ids": ["E001"], "answer": "B is at (99,99)"}])
def test_unverifiable_output_never_enters_memory_and_usage_is_counted(invalid):
    replies = iter([{"support": "partial", "ids": ["E002a"]}, invalid])
    recall = Recall(DOCS, "query", responder(lambda req: response(next(replies), as_json=True)))
    recall.handle("memory_query", {"question": "B?"}, timeout_s=5)
    before = recall.context()["current"]
    result = recall.handle("memory_query", {"question": "Where is A?"}, timeout_s=5)
    assert result["error_code"] == "MODEL_PROTOCOL_ERROR" and recall.context()["current"] == before
    assert recall.stats["requests"] == recall.stats["responses"] == 2
    assert recall.stats["known_usage"]["total_tokens"] == 2 * USAGE["total_tokens"]


def test_insufficient_answer_enables_reobservation_and_transport_failure_has_unknown_usage():
    def handler(req):
        question = json.loads(json.loads(req.content)["messages"][-1]["content"])["question"]
        if question == "Unknown?":
            return response({"support": "insufficient", "ids": []}, usage=None)
        raise httpx.ReadTimeout("do not expose provider details")

    recall = Recall(DOCS, "query", responder(handler))
    assert recall.handle("memory_query", {"question": "Unknown?"}, timeout_s=5)["status"] == "succeeded"
    assert recall.read_ready and recall.context()["current"]["evidence"] == []
    assert recall.handle("memory_query", {"question": "Other?"}, timeout_s=5)["error_code"] == "MODEL_TIMEOUT"
    assert recall.stats["known_usage"] == {} and not recall.stats["usage_complete"]
    assert recall.stats["unknown_usage_requests"] == 2 and recall.stats["responses"] == 1
