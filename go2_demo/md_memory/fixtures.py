"""Explicit deterministic test doubles. None of these results measure an LLM."""

from dataclasses import replace
import re
import numpy as np

from ..robots.go2.profile import GO2_PROFILE
from ..types import SensorFrame, Decision, ToolCall


class FixtureBackend:
    profile = replace(GO2_PROFILE, backend_id="md_memory_fixture", controller_id="synthetic_feedback",
                      display_name="Synthetic interface fixture (no physics)",
                      height_pitch_control=False, posture=None, point_tracking=False,
                      sensor=replace(GO2_PROFILE.sensor, depth_enabled=False, rgb_width_px=64, rgb_height_px=48),
                      simulation_mode="scripted_no_physics", time_mode="synthetic_steps",
                      odometry_source="synthetic_feedback", localization_source="synthetic_feedback")

    def __init__(self):
        self.xy = np.zeros(2)
        self.time_s = 0.
        self.requests = []
        self.closed = False

    def capture(self):
        return SensorFrame(f"fixture-{len(self.requests)}", self.time_s, np.zeros((48, 64, 3), dtype=np.uint8),
                           None, None, None, np.array([*self.xy, 0.]),
                           {"speed_mps": 0., "yaw_rate_radps": 0., "fallen": False},
                           timestamp_s=self.time_s, time_source="synthetic", time_mode="synthetic_steps",
                           localization_available=True, localization_source="synthetic_feedback",
                           depth_available=False, depth_source="unavailable")

    def execute(self, call):
        self.profile.validate_action(call.name, call.arguments)
        self.requests.append(call.name)
        if call.name == "stop":
            return self.stop()
        if call.name != "move_relative":
            raise ValueError("Fixture supports translation and stop only")
        delta = np.array([call.arguments["forward_m"], call.arguments["lateral_m"]])
        self.xy += delta
        self.time_s += 1.
        return {"status": "succeeded", "error_code": None,
                "measured": {"translation_start_frame_m": delta.tolist(), "yaw_delta_rad": 0.,
                             "position_error_m": 0., "source": "synthetic_feedback"}}

    def stop(self):
        return {"status": "succeeded", "error_code": None, "speed_mps": 0., "yaw_rate_radps": 0.}

    def cancel(self):
        return {"status": "succeeded", "cancelled": False, "stop_result": self.stop()}

    def close(self):
        self.closed = True


class FixtureMemoryModel:
    model = "scripted-memory-fixture-not-llm"

    def __init__(self):
        self.counter = 0

    def decide(self, observation, image_data_url, tools, history, **kwargs):
        self.counter += 1
        events = observation["pending_events"]
        notices = [e for e in events if e["kind"] == "user_instruction"]
        changes = []
        if notices:
            event = notices[-1]
            changes = [{"path": "tasks/current_order.md", "body": event["payload"]["notice"],
                        "evidence_ids": [event["id"]], "status": "confirmed"}]
        return Decision([ToolCall(f"fixture-memory-{self.counter}", "commit_memory", {"changes": changes})],
                        model=self.model)

    def close(self):
        pass


class FixturePlannerModel:
    model = "scripted-planner-fixture-not-llm"

    def __init__(self):
        self.turn = 0

    def decide(self, observation, image_data_url, tools, history, **kwargs):
        self.turn += 1
        if self.turn == 1:
            name, args = "memory_read", {"path": "tasks/current_order.md"}
        else:
            document = history[-1]["results"][0]["document"]
            match = re.search(r"ACTIVE R(\d+)[^\n]*?(向左|向右)", document["body"])
            if not match:
                name, args = "abort_plan", {"reason": "No active direction in scripted fixture"}
            else:
                lateral = .25 if match[2] == "向左" else -.25
                name, args = "submit_plan", {"based_on_revision": observation["memory_snapshot"]["revision"],
                    "dependencies": [{"path": document["path"], "sha256": document["sha256"]}],
                    "subgoal": "Execute current recorded short instruction",
                    "completion_condition": "Measured lateral translation agrees with the requested direction",
                    "on_failure": "stop_and_replan", "action_name": "move_relative",
                    "arguments": {"forward_m": 0., "lateral_m": lateral}}
        return Decision([ToolCall(f"fixture-planner-{self.turn}", name, args)], model=self.model)

    def close(self):
        pass
