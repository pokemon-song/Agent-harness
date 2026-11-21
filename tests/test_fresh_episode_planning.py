import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from go2_demo.fresh_episode_planning import CONTACT_RULE, FreshEpisodePlanner, OUTBOUND_PROMPT, public_depth_sectors
from go2_demo.fresh_episode_run import TASK
from go2_demo.paired_return_planning import PAIRED_TOOLS, SYSTEM_PROMPT
from test_paired_return_planning import IMAGE, OBS, ROUTE


@pytest.mark.parametrize("memory_card", [None, {"waypoints": [], "target_evidence": None}])
def test_both_stages_allow_continued_navigation_after_B_contact(memory_card):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        call = {"id": "choice", "type": "function", "function": {
            "name": "follow_waypoints", "arguments": json.dumps(ROUTE)}}
        return httpx.Response(200, json={"choices": [
            {"message": {"tool_calls": [call]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11}})

    model = FreshEpisodePlanner("https://api.deepseek.com", "deepseek-flash", "",
                                transport=httpx.MockTransport(handler))
    history = [{"tool": "scan", "stage": "A", "feedback": f"A-scan-{i}"} for i in range(8)]
    obs = {**OBS, "depth_sectors": {"current_frame_only": True}}
    for stage in ("A", "B"):
        selected = model.choose(obs, IMAGE, memory_card, history, stage=stage, timeout_s=2)
        assert selected.name == "follow_waypoints" and selected.arguments == ROUTE
    outgoing, returning = sent
    system, user = outgoing["messages"]
    assert system["content"] == OUTBOUND_PROMPT
    assert CONTACT_RULE in system["content"] and CONTACT_RULE in TASK
    public = json.loads(user["content"][0]["text"])
    assert [r["feedback"] for r in public["recent_outbound_actions"]] == [f"A-scan-{i}" for i in range(2, 8)]
    assert public["memory_card"] == memory_card
    assert public["observation"] == {"local_odometry": OBS["local_odometry"], "depth_sectors": obs["depth_sectors"]}
    assert user["content"][1]["image_url"]["url"] == IMAGE
    outbound_tools = {t["function"]["name"]: t["function"] for t in outgoing["tools"]}
    assert set(outbound_tools) == {t["function"]["name"] for t in PAIRED_TOOLS}
    assert "蓝球 A" in outbound_tools["follow_waypoints"]["description"]
    assert returning["tools"] == PAIRED_TOOLS
    system, user = returning["messages"]
    assert system["content"] == SYSTEM_PROMPT.replace("碰撞或跌倒会终止任务。", CONTACT_RULE)
    public = json.loads(user["content"][0]["text"])
    assert public["recent_return_actions"] == [] and public["memory_card"] == memory_card
    assert "depth_sectors" not in public["observation"]


def test_public_depth_bands_keep_left_right_distinct_and_missing_unknown():
    depth = np.full((101, 91), np.nan)
    intrinsics = np.array([[100., 0, 45], [0, 100., 50], [0, 0, 1]])
    depth[46:55, :18] = .8
    depth[45:56, 34:56] = 2.
    depth[0, :] = .2  # Far above the camera's narrow near-horizontal band.
    depth[:, 75:] = 20.  # Outside the calibrated working range.
    summary = public_depth_sectors(SimpleNamespace(depth_m=depth, intrinsics=intrinsics))
    assert summary["passage_verified"] is False
    sectors = summary["sectors"]
    assert sectors["left"]["near_p10_z_m"] == .8 and sectors["left"]["bearing_range_deg"][0] > 0
    assert sectors["center"]["closest_z_m"] == 2.
    assert 0 < sectors["left"]["valid_fraction"] < 1
    assert sectors["right"]["status"] == "unknown" and "closest_z_m" not in sectors["right"]
