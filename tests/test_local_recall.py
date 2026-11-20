import json

import httpx
import pytest

from go2_demo.occluded_recall import Recall, RecallResponder


DOCS = {
    "targets/B.md": "[E008] A surface of B was observed.",
    "route/01.md": (
        "[E010] 去程步 0 / f1：实测机身里程计 x=0.000, y=0.000 m，yaw=0.00 rad。\n"
        "[E011] 去程步 12 / f9：实测机身里程计 x=4.100, y=-3.200 m，yaw=1.40 rad。\n"
        "[E012] 去程步 30 / f23：实测机身里程计 x=4.800, y=-2.200 m，yaw=1.50 rad。\n"
        "[E013] 去程步 36 / f29：实测机身里程计 x=5.000, y=-1.200 m，yaw=1.40 rad。"),
    "handoff/end.md": "[E014] 去程步 36 / f29：实测机身里程计 x=5.000, y=-1.200 m，yaw=1.40 rad。",
}
POSE = {"reference": "episode_start_body_horizontal", "translation_m": [5., -1.2], "yaw_rad": 1.4}
ASK = {"question": "Which recent outbound poses and measured cabinet clearance were observed?"}


def test_local_wire_retrieval_cites_nearby_points_and_scopes_cache_to_pose_and_stage():
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        local = json.loads(payload["messages"][1]["content"])
        assert local["documents"] == DOCS and local["question"] == ASK["question"]
        ctx = local["local_context"]
        assert ctx["reference"] == POSE["reference"]
        assert "secret_field" not in ctx
        requests.append(ctx)
        schema = payload["tools"][0]["function"]["parameters"]
        assert schema["properties"]["ids"]["maxItems"] == 3
        answer = {"support": "partial", "ids": ctx["preceding_route_ids_latest_first"],
                  "source_gaps": ["obstacle_clearance"]}
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                            "message": {"content": json.dumps(answer)}}],
                                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}})

    responder = RecallResponder("https://example.invalid", "fake", "", transport=httpx.MockTransport(handler))
    recall = Recall(DOCS, "local-query", responder)
    assert not recall.needs_request("memory_query", ASK)
    assert recall.handle("memory_query", ASK, timeout_s=2)["error_code"] == "MEMORY_LOCATION_REQUIRED"
    assert recall.requests == 0
    recall.set_location({**POSE, "secret_field": "omitted"}, "return")
    assert recall.needs_request("memory_query", ASK)
    assert recall.handle("memory_query", ASK, timeout_s=2)["status"] == "succeeded"
    answer = recall.context()["current"]
    assert requests[-1]["nearest_recorded_route_id"] == "E013"
    assert [item["id"] for item in answer["evidence"]] == ["E012", "E011"]
    assert answer["queried_at"]["stage"] == "return" and "净空" in answer["source_gaps"][0]
    assert not recall.needs_request("memory_query", ASK)
    assert recall.handle("memory_query", ASK, timeout_s=2)["cache_hit"]
    recall.set_location(POSE, "different stage")
    assert recall.needs_request("memory_query", ASK)
    recall.handle("memory_query", ASK, timeout_s=2)
    recall.set_location({**POSE, "translation_m": [4.8, -2.2]}, "different stage")
    assert recall.needs_request("memory_query", ASK)
    recall.handle("memory_query", ASK, timeout_s=2)
    assert requests[-1]["nearest_recorded_route_id"] == "E012"
    assert requests[-1]["preceding_route_ids_latest_first"] == ["E011", "E010"]
    assert recall.stats["requests"] == 3 and recall.stats["cache_hits"] == 1
    assert recall.stats["usage_complete"] and recall.stats["known_usage"]["total_tokens"] == 9


@pytest.mark.parametrize("changes", [
    {"source_gaps": ["this corridor is clear"]}, {"source_gaps": []},
    {"ids": ["E999"]}, {"support": "supported"}, {"source_gaps": ["obstacle_edge"] * 2},
])
def test_rejects_unverified_gaps_or_ids(changes):
    class Responder:
        last_usage, received_response = {}, True

        def select_local(self, question, documents, location, *, timeout_s):
            return {"support": "partial", "ids": ["E008"], "source_gaps": ["obstacle_edge"], **changes}

    recall = Recall(DOCS, "local-query", Responder())
    recall.set_location(POSE, "return")
    result = recall.handle("memory_query", ASK, timeout_s=2)
    assert result["error_code"] == "MODEL_PROTOCOL_ERROR" and not recall.read_ready


@pytest.mark.parametrize("invalid", [{"reference": "world"}, {"translation_m": [1, True]},
                                      {"translation_m": [1]}, {"yaw_rad": float("nan")}])
def test_local_odometry_requires_a_valid_shared_frame(invalid):
    recall = Recall(DOCS, "local-query", object())
    with pytest.raises(ValueError):
        recall.set_location({**POSE, **invalid}, "return")
