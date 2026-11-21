from types import SimpleNamespace

from go2_demo.models import ModelError
from go2_demo.paired_return_memory import PairedReturnMemory


def route(step):
    return (f"[E{step:03}] 去程步 {step} / frame-00001 / t=1.00s：实测机身里程计 "
            f"x={step}.000, y=0.000 m，yaw=0.000 rad；同一去程起点坐标系，x 向前、y 向左。")


def sighting(step, tracked=True):
    kind = "同色区域上的有效 RGB-D 跟踪点 point-001" if tracked else "颜色匹配候选"
    return f"[E1{step:03}] 去程步 {step} / frame-00001 / t=1.00s：目标 B（绿方块）{kind}。"


DOCS = {"route/leg-02.md": route(10), "route/leg-01.md": route(2) + "\n" + route(1),
        "targets/target-B.md": sighting(3) + "\n" + sighting(8) + "\n" + sighting(9, tracked=False)}


class Reader:
    def __init__(self, responses):
        self.calls = []
        self.responses = iter(responses)

    def decide(self, observation, image, tools, history, *, timeout_s):
        self.calls.append((observation, image, history))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        self.raw_usage = {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10, "private": "hidden"}
        return SimpleNamespace(calls=[SimpleNamespace(name="select_route_evidence", arguments=response)])


def ask(slot, budget=1):
    return slot.handle("请给我回程所需的路点和 B 目击。", remaining_requests=budget, timeout_s=10)


def test_empty_and_ledger_use_no_reader_and_ledger_uses_last_valid_sighting():
    empty = PairedReturnMemory("empty", {"invalid": None}, Reader([]))
    assert empty.card is None
    assert ask(empty, 0)["card"] is None
    ledger = PairedReturnMemory("ledger", DOCS, Reader([]))
    assert ledger.card is None
    result = ask(ledger, 0)
    assert result["status"] == "ok" and result["requests"] == 0
    assert [p["id"] for p in result["card"]["waypoints"]] == ["E010", "E002", "E001"]
    assert result["card"]["target_evidence"]["id"] == "E1008"
    result["card"]["waypoints"].clear()
    assert len(ledger.card["waypoints"]) == 3


def test_query_respects_fresh_selection_and_never_reorders_ids():
    pick = {"route_ids": ["E002", "E010"], "target_id": "E1003"}
    reader = Reader([pick, {"route_ids": ["E001"], "target_id": "missing"}])
    q = PairedReturnMemory("query", DOCS, reader)
    bounded = ask(q, 0)
    assert bounded["error"] == "MODEL_REQUEST_BUDGET" and bounded["requests"] == 0 and not reader.calls
    result = ask(q)
    assert result["status"] == "ok" and result["requests"] == 1
    assert [p["id"] for p in q.card["waypoints"]] == pick["route_ids"]
    assert result["known_usage"] == {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}
    assert result["unknown_usage_requests"] == 0 and result["selection"] == pick
    assert reader.calls[0] == ({"stage": "read", "input": {"question": "请给我回程所需的路点和 B 目击。",
                                                            "documents": DOCS}}, None, [])
    bad = ask(q)
    assert bad["error"] == "INVALID_ROUTE_SELECTION" and bad["requests"] == 1
    assert bad["card"] == result["card"] and len(reader.calls) == 2


def test_failed_request_is_charged_and_invalid_question_never_charged():
    reader = Reader([ModelError("timeout", "MODEL_TIMEOUT")])
    q = PairedReturnMemory("query", DOCS, reader)
    assert q.handle("", timeout_s=5, remaining_requests=1)["requests"] == 0
    result = ask(q)
    assert result["error"] == "MODEL_TIMEOUT" and result["requests"] == result["unknown_usage_requests"] == 1
