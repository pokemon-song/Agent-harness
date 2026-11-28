"""Real MuJoCo interface smoke checks with mocked HTTP, never a model benchmark.

The scripted pixel is a geometric probe, not a claim of semantic target detection.
All HTTP requests are intercepted locally; no credentials or network are used.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from go2_demo.general_agent import READER_PROMPT
from go2_demo.general_episode_run import run_episode, write
from go2_demo.general_video import render
from go2_demo.memory_maze_agent import MazeClient


def mock_client_factory():
    decisions = [0]

    def factory(**options):
        reader = options["system_prompt"] == READER_PROMPT
        def handle(request):
            packet = json.loads(request.content)
            content = packet["messages"][-1]["content"]
            data = json.loads(content if isinstance(content, str) else content[0]["text"])
            if reader:
                name = "answer_evidence"
                args = {"answer": "Offline scripted evidence selection; no semantic or task-success claim.",
                        "event_ids": [event["event_id"] for event in data["events"][:2]]}
            else:
                decisions[0] += 1
                obs, state = data["observation"], data["task_state"]
                target_id = data["public_task"]["targets"][0]["target_id"]
                x, y = obs["local_odometry"]["translation_m"]
                number = decisions[0]
                if number == 1:
                    name, args = "set_subgoal", {"description": "Offline measured short move", "criterion": "reach_pose",
                                                  "target_pose": {"x": x + .35, "y": y}}
                elif number == 2:
                    name, args = "follow_waypoints", {"waypoints": [state["active"]["target_pose"]], "after": "inspect",
                                                       "route_role": "primary", "intention": "Exercise real locomotion"}
                elif number in (3, 6, 9):
                    assert state["active"]["status"] == "active", "Measurements must not complete the Agent's subgoal"
                    name, args = "update_subgoal", {"goal_id": state["active"]["id"], "status": "completed",
                        "reason": "Offline scripted judgment from the previous public tool feedback; no task-success claim"}
                    if number == 3:
                        args.update(completed_waypoints=1, route_done=True)
                elif number == 4:
                    name, args = "set_subgoal", {"description": "Offline geometry binding check", "criterion": "observe_target",
                                                 "target_id": target_id, "reason": "Next offline interface check"}
                elif number == 5:
                    name, args = "ground_target", {"target_id": target_id, "image_id": obs["image_id"], "u": 320., "v": 280.}
                elif number == 7:
                    point = next(p for p in obs["grounded_targets"] if p["target_id"] == target_id)
                    name, args = "set_subgoal", {"description": "Verify stopping at the measured surface distance", "criterion": "approach_target",
                                                 "target_id": target_id, "desired_distance_m": point["range_m"]}
                elif number == 8:
                    name, args = "approach_target", {"target_id": target_id, "desired_distance_m": state["active"]["desired_distance_m"],
                                                     "intention": "Verify the real tracked-surface skill without semantic claims"}
                elif number == 10:
                    name, args = "memory_query", {"question": "Which physical actions and measured observations were recorded?"}
                else:
                    name, args = "finish", {"note": "End the offline smoke; this does not claim benchmark completion"}
            return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant",
                "content": None, "tool_calls": [{"id": f"mock-{reader}-{decisions[0]}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}]}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        return MazeClient(base_url="https://offline.invalid/v1", api_key="offline-test", model="offline-test",
                          system_prompt=options["system_prompt"], retain_context=not reader,
                          transport=httpx.MockTransport(handle), trust_env=False)
    return factory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--assets-dir", type=Path, default=Path("assets"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for task in ("occluded_return", "single_object"):
        cfg = SimpleNamespace(task_config=Path("configs/general_agent") / f"{task}.json",
            max_requests=16, max_reader_requests=2, max_consecutive_errors=2, max_wall_seconds=120.,
            max_sim_seconds=120., max_native_actions=24, actions_per_slice=6, mode="query", provider="compatible",
            subgoal_updates="agent",
            model="offline-mock-http", api_key_file=None, assets_dir=args.assets_dir, seed=0, output=args.output / task)
        with patch("go2_demo.general_episode_run.MazeClient.from_env", side_effect=mock_client_factory()):
            result = run_episode(cfg)
        assert result["termination"] == "finish", result
        assert result["success"] is False, "Scripted geometric smoke must not count as target completion"
        assert result["requests"] == {"planner": 11, "reader": 1}, result
        assert result["tools_used"]["update_subgoal"] == 3
        events = [json.loads(line) for line in (cfg.output / "events.jsonl").read_text().splitlines()]
        native = [row for row in events if row["event"] == "native_action"]
        assert any(row["detail"]["tool"] == "move_relative" for row in native)
        ground = next(row["feedback"] for row in events if row["event"] == "feedback" and row["tool"] == "ground_target")
        assert ground.get("geometry_only") is True and ground.get("identity_basis") == "model_proposed", ground
        approach = next(row["feedback"] for row in events if row["event"] == "feedback" and row["tool"] == "approach_target")
        assert approach["status"] == "skill_completed" and approach["reference"] == "tracked_surface_point", approach
        video = render(cfg.output / "capture")
        results.append({"task": task, "passed_interface_check": True, "scripted_not_benchmark": True,
                        "external_api_requests": 0, "mock_requests": result["requests"], "video": video})
    write(args.output / "offline-check.json", {"success": True, "external_api_requests": 0, "checks": results})
    print(json.dumps({"success": True, "external_api_requests": 0, "physical_tasks": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
