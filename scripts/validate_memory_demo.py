"""Offline persistence demo using a scripted test double and public Harness calls.

No API client, credentials, simulator, pretrained policy or evaluator is used.
The scripted decisions are demonstration fixtures, not a production agent mode.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from go2_demo.agent import MemoryAgent
from go2_demo.harness import Harness
from go2_demo.memory import MemoryScope, MemoryStore
from go2_demo.provenance import source_hashes
from go2_demo.robots.go2.profile import GO2_PROFILE
from go2_demo.types import Decision, SensorFrame, ToolCall


LESSON = "measured_travel: 本次请求前进0.30米，公开执行反馈只实测前进0.23米；应使用实测反馈和local_odometry确认进度，不能累加命令距离。"
FIRST_NOTE = "本回合已记录一次公开运动反馈教训；工作笔记仅属于当前回合。"
SECOND_NOTE = "已重新检索并读取此前教训，本回合继续以实测反馈核对移动进度。"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DemoBackend:
    """A small synthetic feedback fixture; it is not a Go2 physics implementation."""

    profile = replace(
        GO2_PROFILE, backend_id="offline_memory_demo", display_name="Go2 interface test double (no physics)",
        controller_id="synthetic_feedback_fixture", height_pitch_control=False, posture=None,
        point_tracking=False, sensor=replace(GO2_PROFILE.sensor, depth_enabled=False, rgb_width_px=30, rgb_height_px=20),
        simulation_mode="scripted_no_physics", time_mode="synthetic_steps",
        odometry_source="synthetic_test_feedback", localization_source="synthetic_test_feedback")

    def __init__(self):
        self.xy = np.zeros(2)
        self.time_s = 0.
        self.frames = 0
        self.motion_requests = []
        self.closed = False

    def capture(self):
        self.frames += 1
        return SensorFrame(
            frame_id=f"synthetic-frame-{self.frames}", sim_time_s=self.time_s,
            rgb=np.zeros((20, 30, 3), dtype=np.uint8), depth_m=None, intrinsics=None,
            camera_to_world=None, base_xy_yaw=np.array([*self.xy, 0.]),
            robot_state={"speed_mps": 0., "fallen": False}, timestamp_s=self.time_s,
            time_source="synthetic", time_mode=self.profile.time_mode, localization_available=True,
            localization_source=self.profile.localization_source, depth_available=False,
            depth_source="unavailable")

    def execute(self, call):
        self.profile.validate_action(call.name, call.arguments)
        if call.name == "stop":
            return self.stop()
        if call.name != "move_relative":
            raise ValueError("This demo fixture only implements move_relative and stop")
        self.motion_requests.append(dict(call.arguments))
        self.xy += np.array([.23, 0.])
        self.time_s += .2
        return {"status": "succeeded", "error_code": None,
                "measured": {"translation_start_frame_m": [.23, 0.], "yaw_delta_rad": 0.,
                             "source": "synthetic_test_feedback"}}

    def stop(self):
        return {"status": "succeeded", "error_code": None, "speed_mps": 0., "yaw_rate_radps": 0.}

    def cancel(self):
        return {"status": "succeeded", "cancelled": False, "reason": "no_active_action", "stop_result": self.stop()}

    def close(self):
        self.closed = True


class DemoModel:
    """Deterministic fixture decisions; no network calls or model reasoning."""

    def __init__(self, *, second_episode=False):
        self.second_episode = second_episode
        self.turn = 0
        self.last_results = []

    def decide(self, observation, image, tools, history, *, timeout_s):
        index = self.turn
        self.turn += 1
        if index == 0:
            name, arguments = (("memory_search", {"query": "measured_travel", "limit": 5}) if self.second_episode
                               else ("move_relative", {"forward_m": .3, "lateral_m": 0.}))
        elif index == 1:
            if self.second_episode:
                matches = self.last_results[0].get("matches", [])
                if not matches:
                    raise RuntimeError("Reopened persistent memory returned no matching lesson")
                name, arguments = "memory_read", {"memory_id": matches[0]["memory_id"]}
            else:
                refs = observation["agent_context"]["public_evidence_refs"]
                if not refs:
                    raise RuntimeError("Motion did not produce an eligible public evidence reference")
                name, arguments = "memory_save", {"text": LESSON, "evidence_refs": [refs[-1]]}
        elif index == 2:
            name, arguments = "memory_working_note", {"text": SECOND_NOTE if self.second_episode else FIRST_NOTE}
        elif index == 3:
            name, arguments = "observe", {}
        else:
            name, arguments = "abort", {"reason": "Offline persistence demonstration complete; no task score claimed."}
        return Decision([ToolCall(f"demo-{int(self.second_episode)}-{index}", name, arguments)], model="scripted-demo-not-llm")


def _episode(store, episode_id, *, second_episode=False):
    backend = DemoBackend()
    harness = Harness(backend, "离线接口演示：核对公开运动反馈，记住实测位移教训。")
    agent = MemoryAgent(harness, store, episode_id)
    model = DemoModel(second_episode=second_episode)
    events, history = [], []
    initial_note = store.info()["working_note"]
    try:
        for turn in range(5):
            observation = agent.observation(remaining_wall_s=30., remaining_calls=5 - turn)
            decision = model.decide(observation, None, agent.tools, history, timeout_s=30.)
            results = agent.dispatch_many(decision.calls)
            if len(results) != 1 or results[0].get("status") != "succeeded":
                raise RuntimeError(f"Offline demo tool failed: {results}")
            evidence_ref = agent.record_public_turn(observation, decision.calls, results)
            model.last_results = results
            event = {"observation": observation, "calls": [{"call_id": call.call_id, "name": call.name,
                      "arguments": call.arguments} for call in decision.calls], "results": results,
                     "public_evidence_ref": evidence_ref}
            events.append(event)
            history.append(event)
            if harness.termination is not None:
                break
        final_note = agent.observation(remaining_wall_s=30., remaining_calls=0)["agent_context"]["memory"]["working_note"]
        expected_note = SECOND_NOTE if second_episode else FIRST_NOTE
        if final_note != expected_note or initial_note != "":
            raise RuntimeError("Working note did not remain episode-local and available during the episode")
        if harness.termination != "abort":
            raise RuntimeError("Offline demonstration must end without claiming task success")
        return {"episode_id": episode_id, "mode": store.info()["mode"], "events": events,
                "initial_working_note": initial_note, "final_working_note": final_note,
                "termination": harness.termination, "synthetic_motion_requests": backend.motion_requests}
    finally:
        backend.stop()
        backend.close()


def run_demo(directory):
    """Create fresh evidence only; both episodes exercise the production memory API."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    database = directory / "memory" / "memory.sqlite3"
    database.parent.mkdir()
    scope = MemoryScope.from_profile(DemoBackend.profile)
    result = {"kind": "offline_memory_persistence_demo", "experiment_protocol": "memory-demo-v1",
              "is_llm_result": False, "api_requests": 0, "model_score": None,
              "simulator_used": False, "scripted_test_double": True, "success": False,
              "created_utc": datetime.now(timezone.utc).isoformat(), "code_sha256": source_hashes(ROOT),
              "validation_script_sha256": sha256(Path(__file__)), "episodes": []}
    try:
        store = MemoryStore(database, scope=scope, mode="read_write", episode_id="offline-first")
        try:
            result["episodes"].append(_episode(store, "offline-first"))
            first_snapshot = store.snapshot()
        finally:
            store.close()
        before_bytes = sha256(database)
        store = MemoryStore(database, scope=scope, mode="read_only", episode_id="offline-reopened")
        try:
            before_logical = store.snapshot()["sha256"]
            result["episodes"].append(_episode(store, "offline-reopened", second_episode=True))
            after_logical = store.snapshot()["sha256"]
        finally:
            store.close()
        after_bytes = sha256(database)
        first, second = result["episodes"]
        saved = first["events"][1]["results"][0]
        reread = second["events"][1]["results"][0]["memory"]
        checks = {"motion_through_harness": first["synthetic_motion_requests"] == [{"forward_m": .3, "lateral_m": 0.}],
                  "saved_from_public_motion_evidence": bool(first["events"][0]["public_evidence_ref"]),
                  "reopened_lesson_matches": reread["memory_id"] == saved["memory_id"] and reread["text"] == LESSON,
                  "readonly_database_bytes_unchanged": before_bytes == after_bytes,
                  "readonly_logical_snapshot_unchanged": first_snapshot["sha256"] == before_logical == after_logical,
                  "episode_local_working_notes": all(ep["initial_working_note"] == "" for ep in (first, second))}
        result.update(success=all(checks.values()), checks=checks,
                      readonly_database_sha256_before=before_bytes, readonly_database_sha256_after=after_bytes,
                      memory_snapshot_sha256_before=before_logical, memory_snapshot_sha256_after=after_logical,
                      scripted_decisions=sum(len(ep["events"]) for ep in result["episodes"]))
        if not result["success"]:
            result["error"] = {"type": "DemoAssertionError", "message": "One or more offline persistence checks failed"}
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    (directory / "validation.json").write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    directory = args.runs_dir / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-memory-demo-" + uuid.uuid4().hex[:6])
    result = run_demo(directory)
    print(json.dumps({"success": result["success"], "api_requests": 0, "is_llm_result": False,
                      "record": str(directory / "validation.json"), "error": result.get("error")}, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
